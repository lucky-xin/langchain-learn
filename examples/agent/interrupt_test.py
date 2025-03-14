import re
import uuid
from typing import TypedDict, Annotated

import streamlit as st
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool, Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from examples.factory.llm import LLMFactory, LLMType


class State(TypedDict):
    messages: Annotated[list, add_messages]


def human_input(state: State):
    human_message = interrupt("human_input")
    print(f"Human input: {human_message}")
    return {
        "messages": [
            {
                "role": "human",
                "content": human_message
            }
        ]
    }


def agent(state: State):
    # Agent logic
    ...


@tool
def replace_sim(phone_model: str):
    """Provides instructions for replacing a SIM card in a specific phone model."""
    instructions = {
        "iPhone 13":
            """
            1. Power off your iPhone 13. 
            2. Locate the SIM tray on the right side of the phone.
            3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray. 
            4. Gently push until the tray pops out.
            5. Remove the old SIM card from the tray and place the new SIM card in the tray.
            6. Insert the SIM tray back into the phone.
            7. Power on your iPhone 13.
            """,
        "Samsung Galaxy S21":
            """
            1. Power off your Samsung Galaxy S21. 
            2. Locate the SIM tray on the top of the phone.
            3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.
            4. Gently push until the tray pops out.
            5. Remove the old SIM card from the tray and place the new SIM card in the tray.
            6. Insert the SIM tray back into the phone.
            7. Power on your Samsung Galaxy S21.
            """,
        "Google Pixel 7":
            """
            1. Power off your Google Pixel 7.
            2. Locate the SIM tray on the left side of the phone.
            3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray. 
            4. Gently push until the tray pops out. 
            5. Remove the old SIM card from the tray and place the new SIM card in the tray, making sure it's aligned correctly.
            6. Insert the SIM tray back into the phone. 
            7. Power on your Google Pixel 7.
            """,

        "OnePlus 10 Pro":
            """
            1. Power off your OnePlus 10 Pro. "
            2. Locate the SIM tray on the left side of the phone. 
            3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.
            4. Gently push until the tray pops out. 
            5. Remove the old SIM card from the tray and place the new SIM card in the tray. 
            6. Insert the SIM tray back into the phone. 
            7. Power on your OnePlus 10 Pro.
            """,
        "Xiaomi 12":
            """
             1. Power off your Xiaomi 12. 
             2. Locate the SIM tray on the left side of the phone. 
             3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.
             4. Gently push until the tray pops out. 
             5. Remove the old SIM card from the tray and place the new SIM card in the tray. 
             6. Insert the SIM tray back into the phone. 
             7. Power on your Xiaomi 12.
            """
    }
    return instructions.get(phone_model, "Sorry, I don't have instructions for that phone model yet.")


def is_uncertain(llm_response: str) -> bool:
    """
    Determines if an LLM response indicates uncertainty or inability to provide accurate information.

    Args:
        llm_response(str): The
    response
    text
    from the LLM
    to
    analyze

    Returns:
    bool: True if uncertainty is detected, False
    otherwise
    """

    # Normalize input text
    normalized_text = llm_response.lower().strip()

    # Direct uncertainty indicators
    basic_uncertainty_terms = {
        "epistemic": [
            "not sure",
            "don't know",
            "do not know",
            "doubt",
            "unable to confirm",
            "unable to determine"
        ],

        "hedging": [
            "might",
            "may",
            "could",
            "possibly",
            "somewhat"
        ],

        "limitation": [
            "limited access",
            "limited information",
            "outside my knowledge"
        ]
    }

    # Complex regex patterns for more nuanced uncertainty expressions
    uncertainty_patterns = [
        # Hedging patterns
        r"I (?:think|believe|suppose|assume|guess|imagine|suspect) (?:that )?(?:it )?might be",

        # Explicit uncertainty
        r"I['']m not (?:entirely|completely|totally|fully|absolutely|quite|really) (?:sure|certain|confident)",

        # Model self-reference
        r"(?:As|Being) (?:an?|the) (?:AI|language model|LLM|assistant)",

        # Knowledge limitations
        r"(?:This|That) (?:is|seems to be) (?:beyond|outside) (?:my|the) (?:scope|capability|knowledge|understanding)",

        # Complexity acknowledgment
        r"(?:This|That|The question|The matter|The issue|The topic) (?:is|seems) (?:too |quite |very |extremely )?complex",

        # Request for clarification
        r"Can you (?:please |kindly )?(?:be more specific|provide more context)"
    ]

    # Disclaimer statements
    disclaimer_patterns = [
        r"I am (?:a large )?(?:language model|AI|artificial intelligence)",
        r"For (?:safety|ethical|security) reasons"
    ]

    # Check basic uncertainty terms
    for category, terms in basic_uncertainty_terms.items():
        if any(term in normalized_text for term in terms):
            return True

    # Check complex uncertainty patterns
    if any(re.search(pattern, llm_response, re.IGNORECASE) for pattern in uncertainty_patterns):
        return True

    # Check disclaimer patterns
    if any(re.search(pattern, llm_response, re.IGNORECASE) for pattern in disclaimer_patterns):
        return True

    # Check for multiple hedging terms in close proximity
    hedging_terms = basic_uncertainty_terms["hedging"]
    hedging_count = sum(1 for term in hedging_terms if term in normalized_text)
    if hedging_count >= 2:  # Multiple hedging terms indicate higher uncertainty
        return True
    return False


class HumanInterventionHandler(BaseCallbackHandler):
    def on_llm_end(self, resp, **kwargs):
        if is_uncertain(resp.generations[0][0].text):
            st.session_state.human_input_requested = True
            st.session_state.pending_human_input = True
            st.session_state.uncertain_response = resp.generations[0][0].text


def process_human_intervention():
    """
    Handle human intervention input and update chat history.
    Pre-populates the form with the LLM's uncertain response.
    """
    if st.session_state.waiting_for_human:
        # Get the stored uncertain response
        initial_response = st.session_state.get("uncertain_response", "")
        print("Initial Response:", initial_response)
        with st.form(key=f"human_intervention_form_{st.session_state.form_id}"):
            st.markdown("""The AI provided this response but wasn't confident.
            Please review and modify as needed:""")
            q = st.text_area(
                "Expert Review:",
                value=initial_response,
                height=200
            )

            # Add guidance for the expert
            st.markdown("""
            Please:
            1. Review the AI's response above
            2. Correct any inaccuracies
            3. Add any missing information
            4. Remove any uncertain language
            """)

            submitted = st.form_submit_button("Submit Expert Review")

            if submitted and q:
                # Add human response to chat history
                st.session_state.messages.append({
                    "role": "human-expert",
                    "content": q
                })
                st.session_state.human_response = q
                st.session_state.waiting_for_human = False
                # Clear the stored uncertain response
                st.session_state.uncertain_response = None
                st.session_state.form_id = str(uuid.uuid4())
                st.rerun()
    return False


def human_intervention(state: State):
    """
    Handle human intervention in the conversation flow.
    Manages the transition between AI uncertainty and human expert input.
    """
    if not st.session_state.waiting_for_human:
        st.session_state.waiting_for_human = True
        return {"messages": [AIMessage(content="Requesting expert review...")]}

    if st.session_state.human_response:
        resp = st.session_state.human_response
        st.session_state.human_response = None
        return {"messages": [HumanMessage(content=resp)]}

    return {"messages": []}


def route_tools(state: State):
    """Modified routing to handle waiting states"""
    if st.session_state.waiting_for_human:
        return "human_intervention"

    if not state["messages"]:
        return END

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        return tool_name

    return END


def chatbot(state: State):
    """
    Main chatbot logic with enhanced uncertainty handling.
    """
    if st.session_state.waiting_for_human:
        return {"messages": []}

    messages = state["messages"]
    resp = llm_with_tools.invoke(messages)

    if is_uncertain(resp.content):
        st.session_state.waiting_for_human = True
        # Store the uncertain resp for the intervention form
        st.session_state.uncertain_response = resp.content
        st.session_state.messages.append({
            "role": "assistant",
            "content": """I'm not completely confident about this resp.
            Requesting expert review..."""
        })
        return {"messages": [AIMessage(content="Requesting expert review...")]}

    return {"messages": [resp]}


# Initialize LLM with the handler
callbacks = [HumanInterventionHandler()]
llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI, callbacks=callbacks
)
llm = llm_factory.create_llm()
# Initialize Google Search tool
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duck_duck_go_search",
    description="Duck Duck Go Search for recent results.",
    func=search.run,
)
llm_with_tools = llm.bind_tools([search_tool, replace_sim])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "waiting_for_human" not in st.session_state:
    st.session_state.waiting_for_human = False
if "human_response" not in st.session_state:
    st.session_state.human_response = None
if "form_id" not in st.session_state:
    st.session_state.form_id = str(uuid.uuid4())

# Initialize memory for LangGraph
memory = MemorySaver()
graph_builder = None
# Create graph builder
if st.session_state.graph is None:
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("duck_duck_go_search", ToolNode([search_tool]))
    graph_builder.add_node("replace_sim", ToolNode([replace_sim]))
    graph_builder.add_node("human_intervention", human_intervention)

    # Add edges
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {
            "duck_duck_go_search": "duck_duck_go_search",
            "replace_sim": "replace_sim",
            "human_intervention": "human_intervention",
            END: END
        }
    )

    graph_builder.add_edge("duck_duck_go_search", "chatbot")
    graph_builder.add_edge("replace_sim", "chatbot")
    graph_builder.add_edge("human_intervention", "chatbot")

    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile(checkpointer=memory)
    # Store the graph in session state
    st.session_state.graph = graph
else:
    graph = st.session_state.graph

# Streamlit UI
st.title("LangGraph Chatbot")
st.image(graph.get_graph().draw_mermaid_png(), caption="LangGraph Visualization", use_container_width=True)

# Custom CSS to style different message types
st.markdown("""
    <style>
    .human-expert {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Process any pending human intervention first
process_human_intervention()

# Chat interface with improved message display
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    # Handle different message types
    if role == "human-expert":
        with st.chat_message("assistant", avatar="👨‍💼"):
            st.markdown(content)
    elif role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:  # assistant
        with st.chat_message("assistant"):
            st.markdown(content)

# Only show chat input if not waiting for human intervention
if not st.session_state.waiting_for_human:
    if q := st.chat_input("What's up?"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        # Process through graph
        try:
            for event in st.session_state.graph.stream(
                    input={"messages": [HumanMessage(content=q)]},
                    config={"configurable": {"thread_id": "1"}}
            ):
                for value in event.values():
                    if value.get("messages"):
                        response = value["messages"][-1].content
                        if not st.session_state.waiting_for_human:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            with st.chat_message("assistant"):
                                st.markdown(response)

                        # Check if we need to stop for human intervention
                        if st.session_state.waiting_for_human:
                            st.rerun()  # Trigger a rerun to show the intervention form
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.waiting_for_human = False  # Reset state in case of error
else:
    st.warning("Please provide expert input above to continue the conversation.")
