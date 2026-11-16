import re
from agents import build_reader_agent , build_search_agent , build_factcheck_agent , writer_chain , critic_chain
from langchain_google_genai.chat_models import GoogleRateLimitError

TARGET_SCORE = 8.5
MAX_ROUNDS = 2

def get_score(feedback) -> float:
    # critic prints "Score: 8.5/10" -> pull 8.5
    match = re.search(r"Score:\s*([\d.]+)", str(feedback))
    return float(match.group(1)) if match else 0


def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}. "
                               f"Include statistics, expert reports, and criticism/risks. Return URLs.")]
    })
    state["search_results"] = search_result['messages'][-1].text

    print("\n search result ",state['search_results'])

    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the 3-4 most relevant URLs (different websites) and scrape each one. "
            f"Keep exact numbers, dates and the source URL for every fact.\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].text

    print("\nscraped content: \n", state['scraped_content'])

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    feedback_notes = ""
    best = {"score": 0, "report": ""}

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            #step 3 - writer chain 
            print("\n"+" ="*50)
            print(f"step 3 - Writer is drafting the report (round {round_no}) ...")
            print("="*50)

            state["report"] = writer_chain.invoke({
                "topic" : topic,
                "research" : research_combined + feedback_notes
            })

            print("\n Report\n",state['report'])

            #step 4 - critic report 
            print("\n"+" ="*50)
            print("step 4 - critic is reviewing the report ")
            print("="*50)

            state["feedback"] = critic_chain.invoke({
                "report":state['report']
            })

            print("\n critic report \n", state['feedback'])

            score = get_score(state["feedback"])
            print(f"\n Round {round_no} score : {score}/10")

            if score > best["score"]:
                best = {"score": score, "report": state["report"]}

            if score >= TARGET_SCORE:
                print("\n Target score reached")
                break

            if round_no == MAX_ROUNDS:
                break   # last round, no need to fact check

            #step 5 - fact checker (only when another round will run)
            print("\n"+" ="*50)
            print("step 5 - Fact checker is verifying claims ...")
            print("="*50)

            factcheck_agent = build_factcheck_agent()
            fc_result = factcheck_agent.invoke({
                "messages": [("user",
                    f"Research:\n{research_combined}\n\nReport:\n{state['report']}"
                )]
            })
            state["fact_check"] = fc_result['messages'][-1].text

            print("\n fact check \n", state["fact_check"])

            # send critic + fact check back to writer for next round
            feedback_notes = (
                f"\n\nPREVIOUS REPORT:\n{state['report']}"
                f"\n\nCRITIC FEEDBACK (fix all of this):\n{state['feedback']}"
                f"\n\nFACT CHECK (fix WRONG, remove UNSUPPORTED):\n{state['fact_check']}"
            )

        except GoogleRateLimitError:
            print("\n Gemini quota hit, stopping with best report so far")
            break

    state["best_score"] = best["score"]
    state["report"] = best["report"]

    with open("final_report.md", "w", encoding="utf-8") as f:
        f.write(best["report"])

    print(f"\n Best score : {best['score']}/10  ->  saved to final_report.md")

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)