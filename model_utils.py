import json

import openai
import time
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import tiktoken             # pip install tiktoken

import numpy as np

# Tokenizer
from utils import remove_stopwords_and_lemmatize

tokenizer = tiktoken.get_encoding("cl100k_base")
MAXTOKENSINHISTORY = 2500

# Get the number of tokens for a string, measured using tiktoken


def getTokenLength(strIn):
    tokens = tokenizer.encode(strIn)
    numTokens = len(tokens)
    return numTokens


def get_best_matched_action_using_sent_transformer(allowed_actions, query, model, topN=5, device="cpu"):
    # print(f"size of allowed_actions: {len(allowed_actions)}")
    if query in allowed_actions:
        return query, [(query, 1.0)]

    query_norm = remove_stopwords_and_lemmatize(text=query,
                                                do_stemming=True,
                                                lemmatize=True)
    query_tokens = set(query_norm.split(" "))
    allowed_actions_filtered = []
    word_sim = []
    for action in allowed_actions:
        action_norm = remove_stopwords_and_lemmatize(text=action,
                                                    do_stemming=True,
                                                    lemmatize=True)
        action_tokens = set(action_norm.split(" "))
        num_common_words = len(
            list((action_tokens).intersection(query_tokens)))
        word_sim.append(-1 * num_common_words)

    indices_actions_sorted_desc_word_sim = np.argsort(word_sim)
    if "cuda" in device:
        max_filtered_actions = 100000
    else:
        max_filtered_actions = 1000

    allowed_actions_filtered = [allowed_actions[ind]
        for ind in indices_actions_sorted_desc_word_sim[:max_filtered_actions]]
    # print(f"actions_sorted_desc_word_sim: {allowed_actions_filtered[0:10]}")
    # print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")

    # Second pass: Use the sentence transformer to find the best-matched action
    action_list_embeddings = model.encode(allowed_actions_filtered)
    query_embeddings = model.encode([query])
    sim = cosine_similarity(
        query_embeddings,
        action_list_embeddings
    )
    max_id = np.argmax(sim)

    # Sort the actions by similarity score
    # First, pack the actions and similarity scores into a list of tuples
    action_sim_tuples = []
    for i in range(len(allowed_actions_filtered)):
        action_sim_tuples.append((allowed_actions_filtered[i], sim[0][i]))
    # Second, sort the list of tuples by the similarity score
    action_sim_tuples.sort(key=lambda x: x[1], reverse=True)
    # Return the top 5 tuples
    topN_action_sim_tuples = action_sim_tuples[:topN]
    # Convert top5 to float so that it can be serialized to JSON
    topN_action_sim_tuples = [(x[0], float(x[1]))
                               for x in topN_action_sim_tuples]
    # print(f"top5_action_sim_tuples: {top5_action_sim_tuples}")

    # print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")

    return allowed_actions_filtered[max_id], topN_action_sim_tuples

def extract_insightID_from_plan(text):
    # Split the text into lines
    lines = text.splitlines()
    for line in lines:
        if line.startswith('@@@'):
            # Extract all numbers from the line
            numbers = re.findall(r'\d+', line)
            return [int(num) for num in numbers]
    return []

def extract_negative_insight(text):
    # Regular expression pattern to find rule IDs and corresponding rule text
    rule_pattern = re.compile(r'(\d+)\. (.+)')
    
    # List to store matched negative rules
    negative_rules = []
    
    # Extract and check rules that contain 'NOT'
    for match in rule_pattern.finditer(text):
        rule_number = int(match.group(1))
        rule_text = match.group(2)
        
        if 'NOT' in rule_text:
            negative_rules.append(f"{rule_number}. {rule_text}")
    
    return negative_rules

def extract_insight_from_summary(text, insightID):
    # Regular expression pattern to find rule IDs and corresponding rule text
    rule_pattern = re.compile(r'(\d+)\. (.+)')
    
    # List to store matched rules
    matched_rules = []
    
    # Extract and check rules matching ruleID
    for match in rule_pattern.finditer(text):
        rule_number = int(match.group(1))
        rule_text = match.group(2)
        
        if rule_number in insightID:
            matched_rules.append(f"{rule_number}. {rule_text}")
    
    return matched_rules
    
def run_chatgpt_query_multi_turn(messages,
                      model_name="gpt-4o-mini",  # pass "gpt4" for more recent model output
                      max_tokens=256,
                      temperature=0.0):
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            print(e)
            print("GPT3 error. Retrying in 10 seconds...")
            time.sleep(2)
    # print('######################### NEW QUERRY #########################')
    print(messages)
    # print('######################### RESPONSE #########################')
    print(response['choices'][0]['message']['content'])
    print('#######################################################################')

    return response

def previous_attempt(previous_actions, previous_observations, previous_rationales = [], use_last_k_memories = 5):
    previous_messages = []
    maxTokensInHistory = MAXTOKENSINHISTORY
    action_observation_history_str = ""
    # Iterate from the last to the 0th index
    for i in range(len(previous_actions) - 1, max(len(previous_actions) - use_last_k_memories, -1) , -1): 
        observation = previous_observations[i]
        action = previous_actions[i]
        # Note we prepend at the start of list
        # So we are prepending in reverse order
        # observation after ith action, selected ith action, generated rationale for ith action
        if i < len(previous_actions) - 1:
            previous_messages[:0] = [{
                "role": "user", "content": f"{observation}\n\nWhat action would you like to do next?"
            }]
        else:
            previous_messages[:0] = [{
                "role": "user", "content": f"{observation}"
            }]

        previous_messages[:0] = [{
            "role": "user", "content": f"Selected action: {action}"
        }]

        rationale = ""
        if previous_rationales:
            rationale = previous_rationales[i]
            previous_messages[:0] = [{
                "role": "assistant", "content": f"{rationale}"
            }]
        
        action_observation_history_str_candidate = f"assistant: {rationale} step: {i}\naction: {action}\nobservation: {observation}\n\n" + action_observation_history_str

        # If we have reached the maximum number of tokens, stop.
        if getTokenLength(action_observation_history_str_candidate) > maxTokensInHistory:
            break
        else:
            action_observation_history_str = action_observation_history_str_candidate
    return previous_messages

def planner_next_subtask(task,
                         current_obs,
                         current_inventory,
                         strategy_summary="",
                         summary = "",
                         previous_actions = [],
                         previous_observations = [],
                         successful_subtasks = [],
                         model="gpt-4o-mini",
                         use_last_k_memories=5,
                         temperature=0.0): 
    
    print("---PLAN NEXT SUBTASK")
    success_subtask_prompt = ""
    if successful_subtasks:
                    #    f"and this is the rationale of the most recent subtask(s): {successful_subtask_rationale[-3:]}. "\
        success_subtask_prompt = f"You have already COMPLETED the following subtask(s): [{', '.join(successful_subtasks)}]. "\
                       f"Given the current state and past achieved subtasks, determine the next subtask that will move you closer to the main task. "\
                       f"Since these are continuous subtasks leading toward completing the main task, your next subtask must follow the logical sequence, "\
                       f"which can be derived from the rationale and the previous tasks mentioned above."\
                       f"Ensure that the next subtask is as small and manageable as the previously completed ones but not repeated. "

    strategy_summary_prompt="",
    if strategy_summary:
        strategy_summary_prompt = \
                    f"Here is the Suggested Strategy including sequence of subtasks provided by ancestor agent."\
                    f"Given the current state and previous completed subtask, use suggested strategy as a reference for deciding your next subtask."\
                    f"If you already reached the end of this strategy, forget this and come up with your own next subtask"\
                    f"\n{strategy_summary}"\
                    
    summary_prompt = ""
    if summary:
        summary_prompt += \
            f"Here is a Summary of Learnings based on your previous attempts which help you to decide your next subtask. However your subtask must follow the order specified in the Main Task and Suggested Strategy" \
            f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"
                        
    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. \n\n" \
                     f"Main task: {task} \n\n" \
                     f"{strategy_summary_prompt}\n"\
                     f"{summary_prompt}\n"\
                     f"As the task is complex, try to break it down into manageable subtasks. " \
                     f"At each step, tell me which subtask you want to achieve (e.g., find something, go somewhere, etc.)," \
                     f"and I will confirm if this subtask is achievable given the current state." \
                     f"\n{success_subtask_prompt}\n" \
                     f"Here is what you currently see: {current_obs} \n" \
                     f"Here is what is currently in your inventory: {current_inventory} \n\n" \
                     f"Below you can see the most recent history of you actions " \
                     f"to help you decide your next subtask: "
                     
    plan_next_subtask_query = \
                     f"If the Main Task requires you to 'focus' on something (OBJ), please write FOCUS ON <OBJ> as the next action. FOCUS is a extremely critical action that can be only used the number of times 'focus' is mentioned in the task description. Using it more than that or inappropiately (such as on a wrong object) will terminate the session and the task will be rendered as incomplete." \
                     f"If you performed an action that requires waiting to see the effect (when you do not see the expected result in observation), please write 'wait' as the next subtask."  \
                     f"Scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. " \
                     f"You need to explain 1./ why you think this subtask is achievable from the current state and 2. how this subtask contributes to the main task. "\
                     f"Format your response as follows: "\
                     f"Write '@@@ I used learning id(s):' as a comma-separated list; the list can be empty if no learnings selected from Summary of Learnings. "\
                     f"Write $$$ followed by the rationale. In it, you must assess the previous successful subtasks (if any) by observing the environment to determine whether it CONTRIBUTE or DOES NOT CONTRIBUTE to the main task. "\
                     f"Then, briefly specify what subtask you want to achieve next, explain why this is achievable and necessary to complete the main task. "\
                     f"For example: '$$$ The previous subtask of looking for a bowl in the fridge DOES NOT CONTRIBUTE to the main task, as the observation shows there is NO BOWL here. From the learning list, there must be a bowl inside the cupboard. We need a bowl to mix colors.'"\
                     f"Then write ### followed by the next subtask (State your next subtask concisely. For example: '### Find bowl.'). \n"\

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]

    if len(previous_actions):
        previous_messages = previous_attempt(previous_actions, previous_observations, use_last_k_memories=use_last_k_memories)
        for prev_message in previous_messages:
            new_messages.append(prev_message)

    new_messages.append({
        "role": "user", "content": f"{plan_next_subtask_query}"
    })

    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str
    # print("RAW RESPONSE STRING:")
    # print(response_str)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # actions should be between "###" blocks. Take the first one (in case it generated multiple ones) as the next action
    temp_str = response_str.split("###")
    reasoningStr = " "
    next_subtask_str = " "
    if len(temp_str) > 1:
        reasoningStr = temp_str[0]
        next_subtask_str = temp_str[1].lower().strip()
    next_subtask_str = next_subtask_str.replace(".", "").replace("i would like to ", "") # .split(' and ')[0]

    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    if len(reasoningStr) < 1:
        reasoningStr = " "
    if (len(next_subtask_str) < 1):
        next_subtask_str = " UNKNOWN "

    response['reasoningStr'] = reasoningStr
    response['next_subtask'] = next_subtask_str

    return response 


def planner_refine_subtask(task,
                           current_obs,
                           current_inventory,
                           current_subtask,
                           current_subtask_rationale,
                           previous_rationales=[],
                           previous_actions=[],
                           previous_observations=[], 
                           model="gpt-4o-mini",
                           use_last_k_memories=5,
                           temperature=0.0):
    
    print("---REFINE SUBTASK")

    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. \n\n" \
                     f"As the task is complex, try to break it down into manageable subtasks. " \
                     f"At each step, I will provide the current unachievable subtask and a list of all trials that have been attempted. " \
                     f"Main task: {task} \n\n" \
                     f"Here is what you currently see: {current_obs}\n" \
                     f"Here is what is currently in your inventory: {current_inventory} \n\n" \
                     f"The following sub-task has been identified: {current_subtask}. "\
                     f"We want to achieve this because: {current_subtask_rationale}\n "\
                     f"However, despite exhaustive trials below, it remains unachievable: \n"\

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]
    if len(previous_actions):
        previous_messages = previous_attempt(previous_actions, previous_observations, previous_rationales, use_last_k_memories)
        for prev_message in previous_messages:
            new_messages.append(prev_message)

    next_action_query = f"Your job is to: " \
                        f"1/ Explain why this sub-task cannot be achieved given the current state and previous trials. " \
                        f"2/ Based on the observation from environment, propose an alternative sub-task contributing to the main task," \
                        f"OR break the current sub-task into smaller, manageable steps." \
                        f"Scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. " \
                        f"Format your response as follows:\n" \
                        f"Write @@@ followed by the brief explanation of why the current subtask cannot be achieved." \
                        f"Then write $$$ followed by the brief rationale including which learning id(s) (if provided) support your point (Explain briefly why this subtask is achievable and necessary to achieve the maintask. For example: '$$$ There must be a bowl in the kitchen. We need a bowl to mix colors.\n')." \
                        f"For example (if learning id(s) are provided): '@@@ There is no bowl in the kitchen. $$$ Using id(s) 2, 3: We need go to art studio to find a bowl.'). "\
                        f"Finally write ### followed by the refined subtask: ' (State your next sub-task concisely. For example: '### Go to art studio.')"

    new_messages.append({
        "role": "user", "content": f"{next_action_query}"
    })

    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str

    # actions should be between "###" blocks. Take the first one (in case it generated multiple ones) as the next action
    temp_str = response_str.split("###")
    reasoningStr = " "
    next_subtask_str = " "
    if len(temp_str) > 1:
        reasoningStr = temp_str[0]
        next_subtask_str = temp_str[1].lower().strip()
    next_subtask_str = next_subtask_str.replace(".", "").replace("i would like to ", "") # .split(' and ')[0]
    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    if len(reasoningStr) < 1:
        reasoningStr = " "
    if (len(next_subtask_str) < 1):
        next_subtask_str = " UNKNOWN "

    response['reasoningStr'] = reasoningStr
    response['next_subtask'] = next_subtask_str

    return response 

def executor_next_action(task,
                        current_obs,
                        current_inventory,
                        objects_set,
                        next_actions_set,
                        previous_actions = [],
                        previous_observations = [],
                        model = "gpt-4o-mini",
                        summary = "",
                        temperature = 0.0,
                        use_last_k_memories=5
                    ):
    # We first ask the model to geberate goal (rationale) and then generate next action
    print("---EXECUTOR FIND ACTION")
    next_action_query = ""

    # Always have task information as first message
    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. " \
                     f"At each step, tell me which action you want to do, e.g., pick up something, " \
                     f"open something, move something to something etc. and I will tell you the result. " \
                     f"Then tell me the next action you want to do, until you complete the task." \
                     f"\n\n" \
                     f"Task: {task}" \
                     f"\n\n"
    if summary:
        sw_prompt_task += \
            f"Here is a summary of learnings based on your previous attempts on this task." \
            f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"
        
    sw_prompt_task += f"Below you can see the most recent history of you actions to help you decide your next action. \n"\

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]

    if len(previous_actions):
        previous_messages = previous_attempt(previous_actions, previous_observations, use_last_k_memories=use_last_k_memories)
        for prev_message in previous_messages:
            new_messages.append(prev_message)

    next_actions_set = [x for x in next_actions_set if 'focus' not in x]
    objects_str = '\n '.join(objects_set)
    actions_str = '\n '.join(next_actions_set)
    # Hints for how to use the actions
    actions_str = actions_str.replace("mix OBJ",
                                      "mix OBJ (here, OBJ should be the container the items to be mixed are in)")

    next_action_query += f"Here is what you currently see:" \
                 f"\n" \
                 f" {current_obs}" \
                 f"Here is what is currently in your inventory:" \
                 f"\n" \
                 f" {current_inventory}" \
                 f"\n\n"

    next_action_query += \
                 f"Possible objects ( value an OBJ can take ) :" \
                 f"\n {objects_str}" \
                 f"\nYour next action should be in one of the following formats:" \
                 f"\nPossible actions ( you should use directly the action in here instead of making process complex ):" \
                 f"\n {actions_str}\n\n" \
                 f"If I say \"Ambiguous request\", your action might mean multiple things. In that case, respond with the number corresponding to the action you want to take.\n" \
                 f"What action would you like to do next?\n"\
                 f"First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. "\
                 f"Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the last observation. " \
                 f"Format your response as follows:\n"\
                 f"Write '@@@ I used learning id(s):' as a comma-separated list; the list can be empty if no learnings selected. "\
                 f"Then write $$$ followed by the rationale."\
                 f"Finally, write ### followed by the SINGLE next action you would like to take."\
                 f"If the task requires you to FOCUS ON something (OBJ), you must write FOCUS ON <OBJ> as the next action. "\
                 f"FOCUS is a extremely critical action that can be only used the number of times 'focus' is mentioned in the task description. Using it more than that or inappropiately (such as on a wrong object) will terminate the session and the task will be rendered as incomplete." \
                 f"If you think you have completed the task, please write TASK_COMPLETE as the next action. " \
                 f"If you performed an action that requires waiting to see the effect, please write 'wait' as the next action."  \

    new_messages.append({
        "role": "user", "content": f"{next_action_query}"
    })

    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )

    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str

    tempStr = response_str.split("###")
    reflectionStr = " "
    reasoningStr = " "
    next_action_str = " "
    if len(tempStr) > 1:
        reasoningStr = tempStr[0]
        next_action_str = tempStr[1].lower().strip()  # The first index should be it's reasoning, the second should be it's action.

    next_action_str = next_action_str.replace(".", "").replace("i would like to ", "") # .split(' and ')[0]
    if (len(next_action_str) < 1):
        next_action_str = " UNKNOWN "

    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    tempStr = reasoningStr.split("@@@")
    if len(tempStr) > 1:
        reflectionStr = tempStr[0].strip()
        reasoningStr = tempStr[1].strip()
    
    if len(reflectionStr) < 1: 
        reflectionStr = " "
    if len(reasoningStr) < 1:
        reasoningStr = " "

    response['pred_next_action'] = next_action_str
    response['reasoningStr'] = reasoningStr
    response['reflectionStr'] = reflectionStr
    return response


def evaluator_action_alignment(task,
                               current_obs,
                               current_inventory,
                               previous_actions,
                               previous_observations,
                               action_candidate, 
                               summary,
                               model="gpt-4o-mini",
                               use_last_k_memories=5,
                               temperature=0.0):
    print("---EVALUATOR ACTION ALIGNMENT")
    sw_prompt_task = f"At each step, I will provide you the task, the current action candidate and list of rules. "\
                     f"Task: {task} \n\n" \
                     f"Below you can see the most recent history of you actions to help you decide your next action. \n"\

    new_messages = [
        {"role": "system",
         "content": "You are an AI evaluator responsible for determining whether a generated action "
                    "violates any given rule. "
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]

    if len(previous_actions):
        previous_messages = previous_attempt(previous_actions, previous_observations, use_last_k_memories=use_last_k_memories)
        for prev_message in previous_messages:
            new_messages.append(prev_message)

    next_action_query = \
                     f"Here is what you currently see: {current_obs} \n" \
                     f"Here is what is currently in your inventory: {current_inventory} \n\n" \
                     f"Rules: {summary} \n"\
                     f"Your task is to assess if the action candidate: '{action_candidate}' violates any rules above (the action should specifically indicating in the rule). \n\n"\
                     f"If any rule mentions 'focus/focusing on OBJ', it specifically indicates this rule is violated if and only if the action is 'focus on OBJ'. "\
                     f"Therefore any action different from 'focus on OBJ' that interacts with that OBJ can be ACCEPTED. "\
                     f"Format your response as follows:\n"\
                     f"If the action does not violate any rule, write '@@@ 1'. "\
                     f"Otherwise, write'@@@ 0' then write '### ' followed by your feedback of which rule it is violated, do NOT indicated the rule number. \n"\

    new_messages.append({
        "role": "user", "content": f"{next_action_query}"
    })

    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str
    tempStr = response_str.split("###")
    approved = (len(tempStr) == 1 or "1" in tempStr[0])
    feedback = " "
    if len(tempStr) > 1:
        feedback = tempStr[1]

    response['approved'] = approved
    response['feedback'] = feedback
    return response


def evaluator_task_complete(task,
                               current_obs,
                               current_inventory,
                               previous_actions,
                               previous_observations, 
                               model="gpt-4o-mini",
                               use_last_k_memories=5,
                               temperature=0.0):
    print("---EVALUATOR TASK COMPLETE")
    sw_prompt_task = f"At each step, I will provide you the primary task, the current sequence of action. "\
                     f"For each action, there is an observation that provide details about the new state of the world."\
                     f"Task: {task} \n\n" \
                     f"Here is what you currently see: {current_obs} \n" \
                     f"Here is what is currently in your inventory: {current_inventory} \n\n" \
                     f"Below is the SEQUENCE OF ACTION:\n\n"
    
    task_complete_prompt = \
                f"Based on the observation trace, your role is to decide if the task is completed or not."\
                f"Format your response as follows:\n"\
                f"Write @@@ followed by your rationale. "\
                f"Then write '### 1' if the task is completed, otherwise write '### 0'"\

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]

    if len(previous_actions):
        previous_messages = previous_attempt(previous_actions, previous_observations, use_last_k_memories=use_last_k_memories)
        for prev_message in previous_messages:
            new_messages.append(prev_message)

    new_messages.append({"role": "user", "content": f"{task_complete_prompt}"})

    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str
    response['feedback'] = response_str.split('###')[0]
    response['isComplete'] = ("### 1" in response_str)
    return response


def evaluator_action_matching(
              action_candidate,
              rationale,
              allowed_actions,
              model="gpt-4o-mini",
              temperature=0.0):
    print("---EVALUATOR ACTION MATCHING")
    sw_prompt_task = f"Your task is to compare the action candidate with the predefined list of possible actions. "\
                     f"The evaluation must focus on strict semantic alignment, not on exact matching. "\
                     f"The predefined action list may be limited and might not fully capture the intended meaning of the action candidate. "\
                     f"While the action candidate itself may seem illogical, the rationale behind it is always logical and must be considered during evaluation. "\
                     f"Action Candidate: {action_candidate}. \n"\
                     f"Rationale: {rationale}. \n"\
                     f"Predefined list: {allowed_actions}. \n"\
                     f"Evaluate whether the action candidate, when supported by the rationale, aligns semantically with any action in the list. "\
                     f"If a match is found, write ### followed by the index of the best-matched action (counting from 0). "\
                     f"If no match is found, write: '### Given the current state, you cannot execute the action: '{action_candidate}'. "\
                     f"There may be intermediate steps necessary before executing the current action candidate OR "\
                     f"this action is already taken and you need to 'wait' to see the effect OR or the action is too trivial to proceed with."\

    new_messages = [
        {"role": "system",
         "content": "You are an AI evaluator helping evaluate a generated action and "
                    "a list of possible actions for an science experiment in a simulated environment."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]
    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str
    response['pred_next_action'] = response_str.replace('###',"").strip()

    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return response 

def success_map(metric, score):
    feedback = ''
    if metric == 'reward':
        if score == -100:
            feedback += "The agent made critical mistakes i.e FOCUS on unapproved object and the task was terminated."
        if score < 0:
            feedback += "The agent performed very poorly and could not make any critical progress."
        if score >= 0 and score < 20:
            feedback += "The agent performed poorly and made some progress but not enough to solve the task."
        if score >= 20 and score < 50:
            feedback += "The agent performed moderately and made some critical progress but not enough to solve the task."
        if score >= 50 and score < 90:
            feedback += "The agent performed very well and made significant critical progress but not enough to solve the task."
        if score >= 90 and score < 100:
            feedback += "The agent performed exceptionally well and made significant critical progress, was just slight away from solving the task."
        if score == 100:
            feedback += "The agent performed exceptionally well and successfully solved the task."
    
    return feedback

def get_trace(data, truncate=False):
    trace = "\n\nCURRENT TRACE\n\n"
    trace += "Task: {}\n\n".format(data["taskDescription"])
    if data['history']:
        for item in data['history']:
            # print(item)
            # trace += "Rationale: {}\n".format(item.get('rationale', ""))
            # trace += "Reflection: {}\n".format(item['reflection'])
            # trace += "Reflection: {}\n".format(item['reflection'])
            # trace += "Rationale: {}\n".format(item['rationale'])
            trace += "Action: {}\n".format(item['action'])
            if truncate:
                trace += "Observation: {}\n\n".format(item['observation'].split('.')[0])
            else:
                trace += "Observation: {}\n\n".format(item['observation'])
                

            # optional cummulative PR

        # we assume if PRF is not computed, we will not have this field
        trace += "\n\nEVALUATION REPORT:\n"
        trace += "REWARD_FINAL: {}. This means: {}\n".format(data['finalScore'], success_map('reward', data['finalScore']))

    return trace

def format_memory(memories):
    # memories list of last-k jsons
    memory_string = "\n\nPREVIOUS LEARNINGS\n\n"
    for m in memories:
        if m['summary']:
            memory_string += "TASK: {}\n".format(m['taskDescription'])
            memory_string += "EPISODE: {}\n".format(m['episodeIdx'])
            memory_string += "LEARNINGS: {}\n".format(m['summary'])

            memory_string += "\nEVALUATION REPORT (for the attempt associated with the learning):\n"
            final_score = m['finalScore']
            memory_string += "REWARD_FINAL: {}. This means: {}\n".format(final_score, success_map('reward', final_score))
            memory_string += '\n'
    
    return memory_string

def summarize(trace, summary_prompt, system_prompt, demo_examples="", prev_memories="", model="gpt-4o-mini", temp=0.5, tokens=1000):
    print(f"trace:{trace}")
    print(f"summary_prompt:{summary_prompt}")
    print(f"system_prompt:{system_prompt}")
    print(f"demo_examples:{demo_examples}")
    print(f"prev_memories:{prev_memories}")

    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": summary_prompt},
                            {"role": "user", "content": demo_examples},
                            {"role": "user", "content": prev_memories},
                            {"role": "user", "content": trace}],
                    temperature=temp, 
                    max_tokens=tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        except Exception as e:
            print(e)
            print("GPT3 error. Retrying in 10 seconds...")
            time.sleep(2)

    output_summary = response["choices"][0]["message"]["content"]
    return output_summary


def summarize_subtask_planning(
                    task,
                    current_run,
                    suggested_trace,
                    model="gpt-4o-mini",
                    temperature=0.5,
                    tokens=1000):
    
    print("---SUMMARIZE PLANNING SUBTASK")

    trace = get_trace(current_run)

    system_prompt = "You are the master strategist, crafting a plan of subtasks to effectively solve the main task."

    task_prompt = \
            f"You are given CURRENT TRACE, a sequence of all actions that the agent made to complete the Task." \
            f"Task: {task} \n"\
            f"For each action, there is an observation that provide details about the new state of the world after each action was executed."\
            f"\n{trace}\n"\
            f"However, not all of the actions are useful. Below are the sequence of action that mainly contribute to the main task."\
            f"{suggested_trace}\n"\
            f"Your role is to create a strategy (which can capture the most useful action), which "\
            f"should be a list of subtasks, with each subtask responsible for a subsequence of actions, followed by a rationale explaining why the agent should perform it. "\
            f"This will save time for the agent in future trials by avoiding exploration of incorrect subtasks and focusing on the ones that contribute. "\
            f"Format your response as follows:\n"\
            f"Write '$$$ ---LEARNING STRATEGY---' followed with your suggested strategy. "\
            f"The strategy is numbered list of subtask, for example the first 2 rows: '1. subtask_1, 2.subtask_2'"
    
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": task_prompt}]
    response = run_chatgpt_query_multi_turn(messages=messages, model_name=model, temperature=temperature, max_tokens=tokens)
    output_startegy = response["choices"][0]["message"]["content"]
    output_startegy = output_startegy.split('$$$')[1]
    return output_startegy 

def summarize_trace_for_preconditions_sTsW(current_run,
                        prev_runs_list=None,
                        model="gpt-4o-mini",
                        temp=0):

    prev_memories = ""
    
    system_prompt = "You are an expert assistant."
    summary_prompt = "You are given CURRENT TRACE, a sequence of actions that an agent made in a world to accomplish a task." \
                            "Task is detailed at the beginning." \
                            "For each action, there is a rationale why the agent made that action.\n" \
                            "There is an observation that provide details about the new state of the world after each action was executed." \
                            "The CURRENT TRACE is accompanied by an EVALUATION REPORT indicating the success of the attempt to the task.\n\n" \
                            "You can also be provided with PREVIOUS LEARNINGS which are learnings from the previous attempts by the agent for the same task in the same environment/world. TASK indicates the task description. EPISODE indicates the number of previous attempts of the task.\n" \
                            "PREVIOUS LEARNINGS also have EVALUATION REPORTs indicated how sucessful the respective attempt was for solving the task."
    
    task_prompt = "Generate a summary of learning, as a numbered list, that will help the agent to successfully accomplish the SAME task AGAIN, in the SAME world.\n" \
            "Each numbered item in the summary can ONLY be of the form:\n" \
                        "X MAY BE NECCESSARY to Y.\n" \
                        "X SHOULD BE NECCESSARY to Y.\n" \
                        "X MAY BE CONTRIBUTE to Y.\n" \
                        "X DOES NOT CONTRIBUTE to Y.\n\n" \
            "Summary of learning as a numbered list:"
    
    final_prompt = summary_prompt + '\n\n' + task_prompt

    # check trace length
    total_budget = 7500 - 1000
    tokens_insturction = getTokenLength(final_prompt)
    total_budget -= tokens_insturction
    trace = get_trace(current_run)
    tokens_trace = getTokenLength(trace)

    memories = {'prev_memory': prev_runs_list, 'trace': trace}

    prev_memories = ""
    if prev_runs_list:
        prev_memories = format_memory(memories['prev_memory'])

    tokens_prev_mem = getTokenLength(prev_memories)

    while tokens_trace + tokens_prev_mem > total_budget:
        if len(memories['prev_memory']) > 0:
            tokens_prev_mem -= getTokenLength(format_memory([memories['prev_memory'][0]]))
            memories['prev_memory'] = memories['prev_memory'][1:]
        else:
            trace = get_trace(current_run, truncate=True)
            tokens_trace = getTokenLength(trace)
            if tokens_trace + tokens_prev_mem > total_budget:
                trace = ' '.join(trace.split(' ')[(total_budget - tokens_trace - tokens_prev_mem):])
            memories['trace'] = trace

    final_trace = memories['trace'] 
    final_prev_memories = ""
    if memories['prev_memory']:
        final_prev_memories = format_memory(memories['prev_memory'])

    summary = summarize(trace=final_trace, summary_prompt=final_prompt, 
                        system_prompt=system_prompt, prev_memories=final_prev_memories,
                        model=model, temp=temp, tokens=1000)

    print("SUMMARY: {}".format(summary))
    return summary
