# 🎬 Multi-Agent Marketing Campaign Generator

A multi-agent AI system built with **AutoGen v0.4+** that automates the creative process of generating marketing campaign ideas, writing video scripts, and reviewing them — all through a team of AI agents that collaborate in a structured conversation.

---

## 📖 Table of Contents

- [Overview](#overview)
- [How It Works — The Big Picture](#how-it-works--the-big-picture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Code Walkthrough](#code-walkthrough)
  - [Part 1: Imports](#part-1-imports)
  - [Part 2: Environment Setup](#part-2-environment-setup)
  - [Part 3: Model Client](#part-3-model-client)
  - [Part 4: Creating the Agents](#part-4-creating-the-agents)
  - [Part 5: Defining the Conversation Flow](#part-5-defining-the-conversation-flow)
  - [Part 6: Creating the Group Chat Team](#part-6-creating-the-group-chat-team)
  - [Part 7: Running the Workflow](#part-7-running-the-workflow)
- [Expected Output](#expected-output)
- [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
- [Key Concepts for Beginners](#key-concepts-for-beginners)

---

## Overview

This project simulates a **marketing creative team** using four AI agents:

| Agent | Role |
|-------|------|
| **Manager** | The boss — gives the brief, evaluates work, makes final decisions |
| **Idea Generator** | The creative mind — brainstorms 3-5 campaign ideas |
| **Script Writer** | The storyteller — writes a full 2-3 minute video script |
| **Script Reviewer** | The quality checker — reviews the script and suggests improvements |

These agents talk to each other in a **fixed sequence**, just like a real creative team would in a meeting.

---

## How It Works — The Big Picture

Imagine a meeting room with four people. The conversation flows like this:

```
Step 1: Manager        → "Here's our product. Give me campaign ideas."
Step 2: Idea Generator → "Here are 5 creative ideas..."
Step 3: Manager        → "I like idea #2. Let's write a script for it."
Step 4: Script Writer  → "Here's the full script with dialogue and scenes..."
Step 5: Manager        → "Script looks good. Let's get it reviewed."
Step 6: Script Reviewer→ "Here's my feedback — strengths, weaknesses, suggestions..."
Step 7: Manager        → "Great work everyone. APPROVED!" (or requests revisions)
```

The word **"APPROVED"** acts as the signal to end the meeting.

---

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key
- Basic familiarity with running Python scripts

---

## Setup

### 1. Install Dependencies

```bash
pip install autogen-agentchat autogen-ext[openai] python-dotenv
```

Here's what each package does:

- **`autogen-agentchat`** — The core AutoGen library for creating AI agents and group chats
- **`autogen-ext[openai]`** — The extension that connects AutoGen to OpenAI's models (GPT-4o-mini, etc.)
- **`python-dotenv`** — A helper that reads your API key from a `.env` file so you don't hardcode secrets

### 2. Create a `.env` File

Create a file named `.env` in the same folder as your script:

```
OPENAI_API_KEY=sk-your-api-key-here
```

> ⚠️ **Never commit your `.env` file to Git!** Add it to your `.gitignore` file.

### 3. Run the Script

```bash
python marketing_campaign_agents.py
```

---

## Code Walkthrough

### Part 1: Imports

```python
# Importing necessary libraries
import os
import asyncio
from dotenv import load_dotenv
```

These are **standard Python utilities**:

- **`os`** — Lets Python interact with your operating system. We use it to read the API key from environment variables. Think of environment variables as secret settings stored on your computer that programs can access.

- **`asyncio`** — Python's built-in library for running **asynchronous** code. Normal Python runs one line at a time and waits. But when our agents call OpenAI's API, they have to wait for a response (sometimes several seconds). `asyncio` lets Python handle these waits efficiently instead of freezing. Think of it like a waiter in a restaurant — instead of standing at one table waiting for customers to decide, the waiter moves to other tables and comes back when the food is ready.

- **`load_dotenv`** from **`dotenv`** — Reads your `.env` file and loads the values (like your API key) into environment variables so your code can access them securely.

---

```python
# Import agent library from autogen
from autogen_agentchat.agents import AssistantAgent
```

**`AssistantAgent`** is the building block for every agent in our system. An `AssistantAgent` is an AI-powered agent that:
- Has a **name** (like "Manager")
- Has a **personality and instructions** (defined by `system_message`)
- Has a **brain** (the LLM model client — GPT-4o-mini in our case)
- Can **read messages** from other agents and **respond** intelligently

Every agent in our team (Manager, Idea Generator, Script Writer, Script Reviewer) is an `AssistantAgent`.

---

```python
# Import group chat and termination conditions
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
```

- **`SelectorGroupChat`** — This is the "meeting room" where all agents come together. It manages the conversation and decides who speaks next. The "Selector" part means it uses a **selector function** (which we write) to pick the next speaker. Without this, the agents wouldn't know when to talk or who goes next.

- **`TextMentionTermination`** — A rule that says "stop the conversation when someone says a specific word." In our case, when the Manager says **"APPROVED"**, the meeting ends. It's like a gavel — when the boss says the magic word, the meeting is over.

- **`MaxMessageTermination`** — A safety net that says "stop after X messages no matter what." This prevents the conversation from running forever (and burning through your API credits) if something goes wrong. We set it to 10 messages.

---

```python
# Import message types
from autogen_agentchat.messages import TextMessage
```

**`TextMessage`** is a message object with two important properties:
- **`content`** — The actual text of the message
- **`source`** — Who sent it (the agent's name)

We use this to create the Manager's opening message. The reason we use `TextMessage` instead of a plain string is critical — if you pass a plain string, AutoGen labels it as coming from `"user"` instead of `"Manager"`, which breaks our turn order. More on this later.

---

```python
# Import Console for streaming output
from autogen_agentchat.ui import Console
```

**`Console`** is a display helper. It takes the stream of messages from the conversation and prints them to your terminal in a nicely formatted way. Without it, you'd have to manually loop through messages and print them yourself.

---

```python
# Import OpenAI client from autogen_ext
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

**`OpenAIChatCompletionClient`** is the bridge between AutoGen and OpenAI. It handles all the technical details of sending prompts to OpenAI's API and receiving responses. You configure it once with your model name and API key, and then every agent shares it.

---

### Part 2: Environment Setup

```python
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

**Line 1: `load_dotenv()`**

This reads your `.env` file (which sits in the same folder as your script) and loads all the key-value pairs into your system's environment variables. So if your `.env` file contains:
```
OPENAI_API_KEY=sk-abc123xyz
```
After `load_dotenv()` runs, that value is now accessible to your Python program.

**Line 2: `os.getenv("OPENAI_API_KEY")`**

This retrieves the value of `OPENAI_API_KEY` from the environment variables and stores it in the Python variable `OPENAI_API_KEY`. If the key doesn't exist in your `.env` file, this returns `None`.

**Why not just paste the API key directly in the code?**

Because it's a security risk. If you push your code to GitHub with the API key hardcoded, anyone can see it and use your key (costing you money). The `.env` file stays on your local machine, and you add it to `.gitignore` so it never gets uploaded.

---

### Part 3: Model Client

```python
# Create the model client
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)
```

This creates a **model client** — the engine that powers all our agents. Think of it as hiring a brain that all four agents share.

- **`model="gpt-4o-mini"`** — Specifies which OpenAI model to use. `gpt-4o-mini` is a fast, cost-effective model that's great for creative tasks. You could swap this for `"gpt-4o"` for higher quality (but higher cost), or `"gpt-3.5-turbo"` for even lower cost.

- **`api_key=OPENAI_API_KEY`** — Passes your API key so the client can authenticate with OpenAI's servers.

We create this **once** and pass it to all four agents. This is efficient — rather than each agent creating its own connection, they all share the same one.

---

### Part 4: Creating the Agents

Each agent is created with three key parameters:

- **`name`** — A unique identifier. Must have **no spaces** (use underscores instead). This name appears in the conversation output and is used by the selector function to pick who speaks next.
- **`description`** — A brief summary of what the agent does. This helps the `SelectorGroupChat` understand each agent's role.
- **`system_message`** — Detailed instructions that define the agent's personality, expertise, and behavior. This is sent to the LLM as a "system prompt" before every response, so the agent always stays in character.
- **`model_client`** — The LLM brain (our shared `model_client`).

#### Manager Agent

```python
manager_agent = AssistantAgent(
    name="Manager",
    description="A marketing campaign manager who provides campaign topics, evaluates ideas and scripts, and makes final decisions.",
    system_message="""You are a marketing campaign manager. Your role is to:
    1. Provide campaign topics to the creative team.
    2. Evaluate the ideas and scripts generated by the team.
    3. Select the best idea and ask the Script Writer to develop it.
    4. Ask the Script Reviewer to review the script.
    5. Make a final decision on whether the script is ready for production.
    
    When you are satisfied with the final script, end your message with the word APPROVED.
    If revisions are needed, specify the key areas to address.
    """,
    model_client=model_client,
)
```

The Manager is the **orchestrator**. Notice the instruction: *"end your message with the word APPROVED"* — this is crucial because our `TextMentionTermination("APPROVED")` condition watches for this word to end the conversation. Without this instruction, the Manager might never say "APPROVED" and the conversation would hit the 10-message safety limit instead of ending gracefully.

The Manager speaks **three times** during the workflow:
1. Opens with the campaign brief
2. Evaluates ideas and picks one for script writing
3. Reviews the final feedback and makes the APPROVED/revision decision

#### Idea Generation Agent

```python
idea_generation_agent = AssistantAgent(
    name="Idea_Generator",
    description="A creative marketing expert who generates innovative campaign ideas, storylines, and characters for video content.",
    system_message="""You are a creative marketing expert. Your role is to:
    1. Generate innovative campaign ideas based on the given topic.
    2. Suggest potential storylines and characters for video content.
    3. Provide a brief outline for each idea.
    Present 3-5 unique ideas for each request.
    """,
    model_client=model_client,
)
```

The Idea Generator is the **brainstormer**. It reads the Manager's brief and generates 3-5 creative campaign ideas. Each idea includes a concept, storyline, and character suggestions.

#### Script Writer Agent

```python
script_writer_agent = AssistantAgent(
    name="Script_Writer",
    description="An expert script writer for marketing videos who develops full scripts with dialogue, scene descriptions, and camera directions.",
    system_message="""You are an expert script writer for marketing videos. Your role is to:
    1. Develop full scripts based on the campaign ideas provided.
    2. Include dialogue, scene descriptions, and camera directions.
    3. Ensure the script aligns with the campaign objectives and target audience.
    Aim for scripts of 2-3 minutes in length.
    """,
    model_client=model_client,
)
```

The Script Writer takes the Manager's selected idea and turns it into a **full production-ready script** with dialogue, scene descriptions, and camera directions.

#### Script Reviewer Agent

```python
script_reviewer_agent = AssistantAgent(
    name="Script_Reviewer",
    description="A critical and detail-oriented script reviewer who analyzes scripts for effectiveness, engagement, and alignment with campaign goals.",
    system_message="""You are a critical and detail-oriented script reviewer. Your role is to:
    1. Analyze scripts for effectiveness, engagement, and alignment with campaign goals.
    2. Provide constructive feedback on dialogue, pacing, and overall structure.
    3. Suggest improvements or alternatives where necessary.
    4. Ensure the script adheres to brand guidelines and target audience preferences.
    Be thorough and specific in your feedback.
    """,
    model_client=model_client,
)
```

The Script Reviewer is the **quality gate**. It analyzes the script and provides structured feedback — strengths, weaknesses, and specific suggestions for improvement.

---

### Part 5: Defining the Conversation Flow

This is the most important part of the code — it controls **who speaks when**.

```python
TURN_ORDER = [
    "Idea_Generator",    # Turn 0: Idea Generator responds to Manager's brief
    "Manager",           # Turn 1: Manager evaluates ideas and picks one
    "Script_Writer",     # Turn 2: Script Writer develops the selected idea
    "Manager",           # Turn 3: Manager forwards script for review
    "Script_Reviewer",   # Turn 4: Script Reviewer provides feedback
    "Manager",           # Turn 5: Manager makes final decision (APPROVED or revisions)
]
```

**`TURN_ORDER`** is a simple Python list that defines the exact sequence of speakers **after** the Manager's opening message. Think of it as an agenda for the meeting:

- Turn 0 → Idea Generator speaks (responds to the Manager's brief)
- Turn 1 → Manager speaks (evaluates the ideas, picks the best one)
- Turn 2 → Script Writer speaks (writes the full script)
- Turn 3 → Manager speaks (passes the script to the reviewer)
- Turn 4 → Script Reviewer speaks (provides feedback)
- Turn 5 → Manager speaks (final verdict — APPROVED or needs revision)

**Why does the list start with Idea Generator and not Manager?**

Because the Manager's opening message is the **task** we pass to `run_stream()`. It's already in the conversation before the selector function is ever called. So the first time the selector is asked "who speaks next?", the Manager has already spoken, and the Idea Generator should go next.

---

```python
def workflow_selector(messages):
    """Custom selector that enforces a fixed conversation sequence."""
    turn = len(messages) - 1
    if turn < len(TURN_ORDER):
        return TURN_ORDER[turn]
    return None
```

This is the **selector function** — the traffic controller of the conversation. Every time an agent finishes speaking, `SelectorGroupChat` calls this function and asks: *"Who should speak next?"*

Here's how it works step by step:

**`messages`** — This parameter receives the full list of all messages in the conversation so far. The first message is always the Manager's opening task.

**`turn = len(messages) - 1`** — We subtract 1 because:
- When the selector is first called, there's 1 message (the Manager's task). `len(messages) = 1`, so `turn = 0`, which maps to `TURN_ORDER[0] = "Idea_Generator"`. ✅
- When called again, there are 2 messages (task + Idea Generator's response). `turn = 1`, which maps to `TURN_ORDER[1] = "Manager"`. ✅
- And so on...

**`if turn < len(TURN_ORDER)`** — Safety check. If we've gone through all 6 turns, don't try to access an index that doesn't exist.

**`return TURN_ORDER[turn]`** — Returns the name of the agent who should speak next.

**`return None`** — If we've exhausted the turn order, return `None` to signal that no one else needs to speak.

Here's a trace of exactly what happens:

| Selector Called | `len(messages)` | `turn` | Returns | What Happens |
|:-:|:-:|:-:|:-:|:-:|
| 1st time | 1 | 0 | `"Idea_Generator"` | Idea Generator brainstorms ideas |
| 2nd time | 2 | 1 | `"Manager"` | Manager picks the best idea |
| 3rd time | 3 | 2 | `"Script_Writer"` | Script Writer writes the script |
| 4th time | 4 | 3 | `"Manager"` | Manager sends script for review |
| 5th time | 5 | 4 | `"Script_Reviewer"` | Script Reviewer gives feedback |
| 6th time | 6 | 5 | `"Manager"` | Manager says APPROVED |

---

### Part 6: Creating the Group Chat Team

```python
# Termination conditions
termination = TextMentionTermination("APPROVED") | MaxMessageTermination(10)
```

This creates the **rules for when the conversation should stop**. The `|` operator means **OR** — the conversation ends if **either** condition is met:

- **`TextMentionTermination("APPROVED")`** — Stop when any message contains the word "APPROVED". This is the **happy path** — the Manager approves the script and the meeting ends cleanly.

- **`MaxMessageTermination(10)`** — Stop after 10 messages total. This is the **safety net** — if the Manager never says "APPROVED" (maybe the LLM forgot the instruction), the conversation won't run forever and rack up API costs.

---

```python
# Create the team
team = SelectorGroupChat(
    participants=[manager_agent, idea_generation_agent, script_writer_agent, script_reviewer_agent],
    model_client=model_client,
    selector_func=workflow_selector,
    termination_condition=termination,
    allow_repeated_speaker=True,
)
```

This assembles the **team** — the meeting room where all agents collaborate.

- **`participants=[...]`** — The list of all agents in the team. Order doesn't matter here since the `selector_func` controls who speaks when.

- **`model_client=model_client`** — The group chat itself needs a model client for internal operations (like understanding the conversation context).

- **`selector_func=workflow_selector`** — Our custom function that controls the turn order. Every time someone finishes speaking, `SelectorGroupChat` calls this function to determine the next speaker.

- **`termination_condition=termination`** — The rules for ending the conversation (APPROVED or 10 messages).

- **`allow_repeated_speaker=True`** — This is **essential**. By default, `SelectorGroupChat` doesn't allow the same agent to speak twice in a row. But in our workflow, the Manager speaks at turns 1, 3, and 5 — sometimes right after another agent, and sometimes the sequence requires it. Without this flag, the system would skip the Manager's turns and break the flow.

---

### Part 7: Running the Workflow

```python
async def main():
    task = TextMessage(
        content=(
            "Hello team, I need creative ideas for our new marketing campaign. "
            "Our product is eco-friendly water bottles. "
            "Could you generate 3-5 innovative campaign ideas, including potential "
            "storylines and characters for video content?"
        ),
        source="Manager",
    )

    result = await Console(team.run_stream(task=task))

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
```

**`async def main():`** — Creates an asynchronous function. We need `async` because the agents make API calls to OpenAI, which take time. The `async/await` pattern lets Python handle these waits efficiently.

**`task = TextMessage(...)`** — This is the **Manager's opening message** that kicks off the entire conversation. Two important parts:

- **`content=(...)`** — The actual text of the message. The parentheses around the strings let you split a long message across multiple lines for readability. Python automatically joins them into one string.

- **`source="Manager"`** — **This is the critical fix.** If you passed a plain string like `team.run_stream(task="Hello team...")`, AutoGen would label the message as coming from `"user"` instead of `"Manager"`. This would break the entire turn order because our `workflow_selector` counts messages to determine turns, and an unexpected `"user"` message would throw off the count. By using `TextMessage` with `source="Manager"`, we ensure the first message is correctly attributed.

**`result = await Console(team.run_stream(task=task))`** — Three things happening here (from inside out):

1. **`team.run_stream(task=task)`** — Starts the conversation. The `run_stream` method returns messages **one at a time** as they're generated (streaming), rather than waiting for the entire conversation to finish.

2. **`Console(...)`** — Takes the stream of messages and prints each one to your terminal in a formatted way with headers like `---------- TextMessage (Manager) ----------`.

3. **`await`** — Pauses execution and waits for the entire conversation to complete. You can only use `await` inside an `async def` function. Without it, the function would return immediately without waiting for the agents to finish talking.

**`result`** — After the conversation ends, this variable contains the complete result including all messages and metadata.

**The print statements** — Simply display a visual separator to show the workflow has completed:
```
============================================================
WORKFLOW COMPLETE
============================================================
```

---

```python
if __name__ == "__main__":
    asyncio.run(main())
```

This is the **entry point** of the script — the very first thing that runs.

- **`if __name__ == "__main__":`** — A standard Python pattern. It means "only run this code if the file is executed directly (not imported as a module)." If someone did `import marketing_campaign_agents` in another file, this block would NOT run.

- **`asyncio.run(main())`** — Since `main()` is an `async` function, you can't call it like a normal function. `asyncio.run()` starts Python's async engine (called the "event loop"), runs `main()` to completion, and then shuts everything down cleanly. Think of it as the **ignition key** that starts the async machinery.

---

## Expected Output

When you run the script, you should see something like this:

```
---------- TextMessage (Manager) ----------
Hello team, I need creative ideas for our new marketing campaign.
Our product is eco-friendly water bottles. Could you generate 3-5
innovative campaign ideas, including potential storylines and
characters for video content?

---------- TextMessage (Idea_Generator) ----------
Here are 5 innovative campaign ideas for eco-friendly water bottles:

1. **"The Ripple Effect"** ...
2. **"Bottle Your Future"** ...
3. **"Refill Revolution"** ...
...

---------- TextMessage (Manager) ----------
Great ideas! I'd like to go with idea #2 "Bottle Your Future."
Script Writer, please develop a full script for this concept...

---------- TextMessage (Script_Writer) ----------
# BOTTLE YOUR FUTURE — Marketing Video Script

SCENE 1: OPENING
[Camera: Wide shot of a landfill...]
...

---------- TextMessage (Manager) ----------
Excellent script! Script Reviewer, please review this script for
effectiveness and alignment with our eco-friendly campaign goals...

---------- TextMessage (Script_Reviewer) ----------
## Script Review: "Bottle Your Future"

**Strengths:**
- Strong emotional hook in the opening scene...

**Areas for Improvement:**
- The transition between scenes 2 and 3 feels abrupt...

**Suggestions:**
...

---------- TextMessage (Manager) ----------
Thank you for the thorough review. The script is strong with minor
adjustments needed. Overall, this is ready for production. APPROVED

============================================================
WORKFLOW COMPLETE
============================================================
```

---

## Common Issues and Troubleshooting

### "ModuleNotFoundError: No module named 'autogen_agentchat'"
You need to install the packages:
```bash
pip install autogen-agentchat autogen-ext[openai] python-dotenv
```

### Messages show "user" instead of "Manager"
Make sure you're passing a `TextMessage` object with `source="Manager"` as the task, not a plain string.

### Conversation never ends / hits 10 message limit
The Manager's `system_message` must include the instruction to say "APPROVED". If the LLM ignores this instruction, increase `MaxMessageTermination` or make the instruction more emphatic.

### "API key not found" error
Make sure your `.env` file is in the same directory as your script and contains:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Agents speaking out of order
Check that agent names in `TURN_ORDER` exactly match the `name` parameter of each agent (case-sensitive, with underscores).

---

## Key Concepts for Beginners

### What is an "Agent"?
An agent is an AI entity with a specific role and personality. It's like giving ChatGPT a character to play. Each agent has instructions (`system_message`) that tell it how to behave, and a brain (`model_client`) that powers its responses.

### What is a "Group Chat"?
A group chat is a structured conversation between multiple agents. Instead of you chatting with one AI, multiple AIs chat with each other (and optionally with you). The `SelectorGroupChat` manages who speaks when.

### What is "async/await"?
When your code needs to wait for something slow (like an API call), `async/await` lets Python do other work while waiting instead of freezing. It's essential for AutoGen because agents make many API calls during a conversation.

### What is a "Selector Function"?
A function you write that tells the group chat who should speak next. It receives all messages so far and returns the name of the next speaker. This gives you full control over the conversation flow.

### What is a "Termination Condition"?
A rule that tells the group chat when to stop. Without it, agents would keep talking forever. Common conditions include watching for a keyword ("APPROVED") or setting a maximum message count.

---

## Project Structure

```
project/
├── marketing_campaign_agents.py   # Main script
└── README.md                      # This file
```


## License

This project is for educational purposes.
