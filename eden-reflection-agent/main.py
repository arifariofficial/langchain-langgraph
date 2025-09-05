from typing import Sequence
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"

class State(BaseModel):
    messages: Sequence[BaseMessage]


def generation_node(state: State):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: State) -> dict:
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(State)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: State):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)