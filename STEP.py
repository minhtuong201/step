import time
import random
import argparse
import os
import json
import torch
import json

from scienceworld import ScienceWorldEnv

from model_utils import get_best_matched_action_using_sent_transformer,\
    extract_insightID_from_plan,\
    extract_negative_insight, \
    extract_insight_from_summary, \
    planner_next_subtask, \
    planner_refine_subtask, \
    executor_next_action, \
    evaluator_task_complete, \
    evaluator_action_alignment,\
    evaluator_action_matching,\
    summarize_trace_for_preconditions_sTsW, \
    summarize_subtask_planning

from sentence_transformers import SentenceTransformer

def task_complete_status(task):
    # List of possible strings that indicate task completion
    completion_phrases = [
        "TASK_COMPLETE",
        "TASK COMPLETE",
        "TASKCOMPLETE",
        "task_complete",
        "task complete",
        "successfully completed",
        "There is no further action required"
    ]
    return any(phrase in task for phrase in completion_phrases)

def stepAgent(args):
    """ STEP agent """

    # Set-up
    task_num = int(args['task_num'])
    var_num = int(args["var_num"])
    print(f"Running STEP agent for taskIdx:{task_num}, varIdx:{var_num}")
    
    sent_transformer_model = SentenceTransformer('bert-base-nli-mean-tokens', device=args['device'])
    use_last_k_memories = args['use_last_k_memories']
    simplificationStr = args['simplification_str']
    numEpisodes = args['num_episodes']
    gpt_model = args['gpt_model']
    temperature = args['temperature']
    summarize_end_of_episode = bool(args['summarize_end_of_episode'])
    env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])

    exitCommands = ["quit", "exit"]

    # Keep track of the agent score
    score_all_eps = []
    finalScores = []
    memory_of_runHistories = []
    output_dir = args['output_path_prefix']
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    summaryfname = args['output_path_prefix'] + "summary.txt"
    summaryFile = open(summaryfname, "w")
    strategy_summary_str = ""
    taskNames = env.getTaskNames()
    print("Task Names: " + str(taskNames))

    for taskIdx in [task_num]:
        # Choose task
        taskName = taskNames[taskIdx]        # Just get first task
        env.load(taskName, 0, "")  
        print("Task Name: " + taskName)
        print("Task Description: " + str(env.getTaskDescription()))

        for varIdx in [var_num]:
            prev_episode_summary_str = ""
            # Start running episodes
            for episodeIdx in range(0, numEpisodes):

                suggested_trace = []
                score_per_eps = []

                # Save history -- and when we reach maxPerFile, export them to file
                filenameOutPrefix = f"{output_dir}/Task{taskIdx}_Var{varIdx}_Ep{episodeIdx}_runhistories.json"
                historyOutJSON = open(filenameOutPrefix, "w")

                # Environment loading
                env.load(taskName, varIdx, simplificationStr, generateGoldPath=True)
                print("Task Name: " + taskName)
                print("Task Variation: " + str(varIdx))
                print("Task Description: " + str(env.getTaskDescription()) )

                # Score = -100 if agent focus on the unapproved object
                score = 0.0
                score_positive = 0.0
                isCompleted = False
                mainStep = 0

                # History of running entire episode
                rawActionHistory = [""]
                rationaleHistory = [""]
                observationHistory=[""]
                subgoalHistory = [env.getGoalProgressJSON()]
                observation, reward, isCompleted, info = env.step("look around")
                score_per_eps.append(info['score'])

                successful_subtask = []
                unsuccessful_subtasks = []
                successful_subtask_rationale = []
                unsuccessful_subtask_rationale = []

                # History for each subtask only. We could generate insight from this
                previous_rationales = []
                previous_actions = []
                previous_observations = []

                generated_subtask = ""
                earlyStop = False
                refined_flag = False

                while (isCompleted == False and generated_subtask not in exitCommands):

                    generated_action_str = ""        # First action
                    generated_rationale_str = ""        # First action
                    best_match_action = generated_action_str

                    # max successful = 16, max unsuccessful = 8
                    if len(successful_subtask) > 16 or len(unsuccessful_subtasks) > 8: 
                        print("Loop detected!")
                        break
                    subStep = 0
                    summary_str = ""
                    if summarize_end_of_episode:
                        summary_str = prev_episode_summary_str

                    ### Plan Next Subtask 
                    if refined_flag == False:
                        response = planner_next_subtask(
                            task=env.getTaskDescription(),
                            current_obs=env.look(),
                            current_inventory=env.inventory(),
                            strategy_summary=strategy_summary_str,
                            summary=summary_str,
                            previous_actions=previous_actions,
                            previous_observations=previous_observations,
                            successful_subtasks=successful_subtask,
                            model = gpt_model,
                            use_last_k_memories=use_last_k_memories
                        )

                    ### Refine Subtask 
                    else:
                        response = planner_refine_subtask(
                            task=env.getTaskDescription(),
                            current_obs=env.look(),
                            current_inventory=env.inventory(),
                            current_subtask=generated_subtask,
                            current_subtask_rationale=generated_subtask_rationale,
                            previous_rationales=previous_rationales,
                            previous_actions=previous_actions,
                            previous_observations=previous_observations,
                            model = gpt_model,
                            use_last_k_memories=use_last_k_memories
                        )
                        unsuccessful_subtasks.append(generated_subtask)
                        unsuccessful_subtask_rationale.append(generated_subtask_rationale)
                        refined_flag = False
                        
                    not_achievable_actions = []
                    not_achievable_feedback = []            
                    generated_subtask_rationale = response['reasoningStr']
                    relevant_insight_ID = extract_insightID_from_plan(generated_subtask_rationale)
                    generated_subtask = response['next_subtask']
                    subtaskComplete = False

                    allow_last_k_action = 0
                    if task_complete_status(generated_subtask):
                        generated_subtask = "exit"
                        print("\n---GENERATED SUBATSK: TASK_COMPLETE!---\n")
                        break

                    while (generated_action_str not in exitCommands):
                        # Number of step in current subtask
                        subStep += 1
                        if best_match_action not in exitCommands and best_match_action != "":
                            observation, reward, isCompleted, info = env.step(best_match_action)
                            # Total interaction with environment
                            mainStep += 1
                            # print('#############')
                            print ("\nMain step: " + str(mainStep))

                            score = info['score']
                            if score > 0.0:
                                score_positive = score

                            # Store subgoal progress
                            subgoalHistory.append(env.getGoalProgressJSON())
                            rawActionHistory.append(best_match_action)
                            rationaleHistory.append(generated_rationale_str)
                            observationHistory.append(observation)
                            
                            print("\n>>> " + observation)
                            print("Reward: " + str(reward))
                            print("Score: " + str(score))
                            print("isCompleted: " + str(isCompleted))

                            score_per_eps.append(score_positive)
                        # The Env will make isCompleted `True` when a stop condition has happened, or the maximum number of steps is reached.
                        if (isCompleted):
                            break

                        summary_str = ""
                        if summarize_end_of_episode:
                            summary_str = prev_episode_summary_str

                        generated_action_str = "N/A"
                        num_retries = 0
                        max_substep = 8 
                        max_last_k_action = 1
                        max_num_retries_executor = 3
                        
                        not_approved_actions = []
                        not_approved_feedback = []

                        relevant_insight = extract_insight_from_summary(summary_str, relevant_insight_ID)
                        previous_observations = list(observationHistory)
                        previous_rationales = list(rationaleHistory)
                        previous_actions = list(rawActionHistory)

                        ### Self-Refinement 
                        while (num_retries < max_num_retries_executor):
                            num_retries += 1

                            # Generated best action 
                            executor_response = executor_next_action(
                                    task=generated_subtask,
                                    current_obs=env.look(),
                                    current_inventory=env.inventory(),
                                    objects_set=env.getPossibleObjects(),
                                    next_actions_set=env.getPossibleActions(),
                                    previous_actions=previous_actions + not_approved_actions + not_achievable_actions,
                                    previous_observations=previous_observations + not_approved_feedback + not_achievable_feedback,
                                    model=gpt_model,
                                    summary=relevant_insight,
                                    temperature=temperature,
                                    use_last_k_memories=use_last_k_memories
                                )
                            generated_rationale_str = executor_response['reasoningStr']
                            generated_action_str = executor_response["pred_next_action"].lower().strip()

                            rules = extract_negative_insight(summary_str)

                            if task_complete_status(generated_action_str) or not rules: break
                            evaluator_response = evaluator_action_alignment(
                                    task=generated_subtask,
                                    current_obs=env.look(),
                                    current_inventory=env.inventory(),
                                    previous_actions=previous_actions,
                                    previous_observations=previous_observations,
                                    action_candidate=generated_action_str,
                                    summary=rules,
                                    model=gpt_model,
                                    use_last_k_memories=use_last_k_memories
                            )
                            if evaluator_response['approved']: break
                            not_approved_feedback.append(f"The generated action: '{generated_action_str}' is not approved. {evaluator_response['feedback']}")
                            not_approved_actions.append(generated_action_str)

                        not_approved_feedback = [] 
                        not_approved_actions = []

                        ### Task complete verification
                        if task_complete_status(generated_action_str) or \
                            ('focus' in generated_action_str \
                                and previous_actions[-1] == generated_action_str):
                            subtaskComplete = True
                            allow_last_k_action += 1
                        if previous_actions and not subtaskComplete and generated_subtask != env.getTaskDescription():            
                            response = evaluator_task_complete(
                                task = generated_subtask,
                                current_obs=env.look(),
                                current_inventory=env.inventory(),
                                previous_actions=previous_actions,
                                previous_observations=previous_observations,
                                model=gpt_model,
                            )
                            subtaskComplete = response['isComplete']
                        if subtaskComplete == True:
                            allow_last_k_action += 1
                            if allow_last_k_action > max_last_k_action:
                                generated_action_str = "exit"
                                best_match_action = generated_action_str
                                topN = []
                                successful_subtask.append(generated_subtask)
                                successful_subtask_rationale.append(generated_subtask_rationale)
                                break

                        if not_achievable_actions.count(generated_action_str) == 3:
                            print(f"\nSame generated action: {generated_action_str} appear more than 3 times in NOT ACHIEVABLE LIST. Exit!")
                            refined_flag = True
                            break

                        if (generated_action_str != "exit"):
                            if (subStep == max_substep):
                                print("Subtask take too long to finish!")
                                earlyStop = True
                                refined_flag = True
                                break

                        if allow_last_k_action <= max_last_k_action:
                            valid_actions_list = info['valid']  # populated in the return from 'step'
                            valid_actions_list = [x for x in valid_actions_list if 'reset' not in x] # remove reset from valid actions

                            if "FOCUS" in executor_response["pred_next_action"] or  "focus" in executor_response["pred_next_action"]:
                                valid_actions_list = [x for x in valid_actions_list if 'focus' in x]
                            else:
                                valid_actions_list = [x for x in valid_actions_list if 'focus' not in x]
                            
                            best_match_score = 0.0
                            best_match_action = "exit"
                            if len(valid_actions_list) == 0:
                                # check "Ambiguous request in observation"
                                if "Ambiguous request" in observation:
                                    valid_actions_list = [str(x) for x in range(len(observation.split('\n')[1:]))]

                            ### Get best match action w/ Sentence Transformer 
                            if len(valid_actions_list) > 0:
                                best_match_action, topN = get_best_matched_action_using_sent_transformer(
                                    allowed_actions=valid_actions_list,
                                    query=generated_action_str,
                                    model=sent_transformer_model,
                                    topN=10,
                                    device=args['device']
                                )

                                # Check top-1 action match score, and if the score < threshold then 
                                best_match_score = topN[0][1]

                            if best_match_score == 1 or \
                                best_match_action in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'] or (num_retries==max_num_retries_executor) or (len(valid_actions_list) == 0) :
                                continue
                            
                            ### Get best match action from list above w/ Evaluator 
                            else:
                                topN_action=[action for action, score in topN]
                                matching_response = evaluator_action_matching(
                                                        action_candidate=generated_action_str,
                                                        rationale=generated_rationale_str,
                                                        allowed_actions=topN_action, 
                                                        model=gpt_model
                                                        )
                                if matching_response['pred_next_action'].isdigit():
                                    index = int(matching_response['pred_next_action'])
                                    best_match_action = topN[index][0]
                                    not_achievable_actions = []
                                    not_achievable_feedback = []    
                                else:
                                    best_match_action = ""
                                    not_achievable_feedback.append(matching_response['pred_next_action'])
                                    not_achievable_actions.append(generated_action_str)
                                    print("ACTION IS NOT ACHIEVABLE!")

                    if (len(previous_actions) > 5):
                        # Check if the last 5 actions were all the same.  If so, exit
                        if (previous_actions[-1] == previous_actions[-2] == previous_actions[-3] == previous_actions[-4] == previous_actions[-5]):
                            print("Last 2 actions were the same.  Exiting.")
                            refined_flag = True
                            break

                        # Check if the max num steps (here, 1.5*gold_sequence_length) have been executed.  If so, exit
                        if len(previous_actions) >= 1.5*len(env.getGoldActionSequence()):
                            print("Model generated an action sequence which is 1.5 times longer than"
                                "the gold action sequence without succeeding at the task.  Exiting.")
                            refined_flag = True
                            break

                print("Goal Progress:")
                print(env.getGoalProgressStr())
                time.sleep(1)

                # Episode finished -- Record the final score
                finalScores.append({
                    "taskIdx": taskIdx,
                    "taskName": taskName,
                    "variationIdx": varIdx,
                    "episodeIdx": episodeIdx,
                    "final_score": score_positive,
                    "isCompleted": isCompleted
                })

                # Report progress of model
                print ("Final score: " + str(score))
                print ("isCompleted: " + str(isCompleted))

                # Show gold path
                gold_path = str(env.getGoldActionSequence())
                print("Gold Path:" + gold_path)

                # Get run history
                runHistory = env.getRunHistory()
                print("############## RunHistory ###############")
                print(runHistory)
                # Store last step
                subgoalHistory.append(env.getGoalProgressJSON())
                rationaleHistory.append(generated_rationale_str)
                rawActionHistory.append(generated_action_str)
                # Add rationales to run history
                for idx, rationaleStr in enumerate(rationaleHistory):
                    if idx >= len(runHistory['history']):
                        break
                    runHistory['history'][idx]['rationale'] = rationaleStr
                    runHistory['history'][idx]['rawAction'] = rawActionHistory[idx]
                    runHistory['history'][idx]['subgoalProgress'] = subgoalHistory[idx]


                # Also store final score
                runHistory['episodeIdx'] = episodeIdx
                runHistory['finalScore'] = score
                runHistory['finalScorePositive'] = score_positive
                runHistory['isCompleted'] = isCompleted
                runHistory['earlyStop'] = earlyStop
                runHistory['model'] = gpt_model
                runHistory['gold-action-seq'] = gold_path
                runHistory['memory-seen'] = summary_str
                
                print(f"\n------ REWARD PER STEP -------\n{score_per_eps}")
                score_all_eps.append(score_per_eps)
                strategy_summary_str = summarize_subtask_planning(
                    task=env.getTaskDescription(),
                    current_run=runHistory,
                    suggested_trace=suggested_trace,
                    model=gpt_model,
                    )
                runHistory['strategy_summary'] = strategy_summary_str

                episode_summary_str = ""
                if summarize_end_of_episode:
                    prev_runs_list = memory_of_runHistories[-3:] # Last 3 episodes
                    episode_summary_str = summarize_trace_for_preconditions_sTsW(runHistory,
                                                        prev_runs_list=prev_runs_list,
                                                        model=gpt_model,
                                                        temp=temperature) # q2 summary passed as gold summary 
                runHistory['summary'] = episode_summary_str
                prev_episode_summary_str = episode_summary_str

                # Save runHistories into a JSON file
                print ("Writing history file: " + filenameOutPrefix)
                memory_of_runHistories.append(runHistory)
                json.dump(runHistory, historyOutJSON, indent=4, sort_keys=False)
                historyOutJSON.flush()
                historyOutJSON.close()

    # Show final episode scores to user:
    print("")
    print(f"\n------ REWARD ALL EPSISODES -------\n{score_all_eps}")
    print("---------------------------------------------------------------------")
    print(" Summary (ChatGPT Agent)")
    print(" Simplifications: " + str(simplificationStr))
    print("---------------------------------------------------------------------")
    print(f"task\tName\tvar\tepi\tscore\tcomplete?")
    for finalScore in finalScores:
        print(f"{finalScore['taskIdx']}\t{finalScore['taskName']}\t{finalScore['variationIdx']}\t"
              f"{finalScore['episodeIdx']}\t{finalScore['final_score']}\t{finalScore['isCompleted']}")
    print("---------------------------------------------------------------------")
    print("")

    print("Completed.")

    summaryFile.write(f"" + '\n')
    summaryFile.write("---------------------------------------------------------------------" + '\n')
    summaryFile.write(" Summary (ChatGPT Agent)" + '\n')
    summaryFile.write(" Simplifications: " + str(simplificationStr) + '\n')
    summaryFile.write("---------------------------------------------------------------------" + '\n')
    summaryFile.write(f"task\tName\tvar\tepi\tscore\tcomplete?" + '\n')
    for finalScore in finalScores:
        summaryFile.write(f"{finalScore['taskIdx']}\t{finalScore['taskName']}\t{finalScore['variationIdx']}\t"
              f"{finalScore['episodeIdx']}\t{finalScore['final_score']}\t{finalScore['isCompleted']}" + '\n')
    summaryFile.write("---------------------------------------------------------------------" + '\n')
    summaryFile.write("Completed." + '\n')
    summaryFile.close()

def build_simplification_str(args):
    """ Build simplification_str from args. """
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")

    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")

    if args["open_containers"]:
        simplifications.append("openContainers")

    if args["open_doors"]:
        simplifications.append("openDoors")

    if args["no_electrical"]:
        simplifications.append("noElectricalAction")

    return args["simplifications_preset"] or ",".join(simplifications)

def parse_args():
    desc = "Run a model that chooses random actions until successfully reaching the goal."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")
    parser.add_argument("--task-num", type=str, default="4",
                        help="Specify the task number to play. Default: %(default)s")
    parser.add_argument("--var-num", type=int, default=1,
                        help="Specify the task variation number to play. Default: %(default)s")
    parser.add_argument("--env-step-limit", type=int, default=100,
                        help="Maximum number of steps per episode. Default: %(default)s")
    parser.add_argument("--num-episodes", type=int, default=2,
                        help="Number of episodes to play. Default: %(default)s")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o-mini",
                        help="Choose GPT model to use ['gpt-3.5-turbo', 'gpt-4o-mini']. Default: %(default)s")
    parser.add_argument("--summarize_end_of_episode", type=int, default=1,
                        help="Summarize at the end of episode (for preconditions)")
    parser.add_argument("--device", type=str, required=True,
                        help="Select device to be used by sentence transformer. ['cpu', 'cuda', 'cuda:0']")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Select temperature for running chatgpt completion api")
    parser.add_argument("--use-last-k-memories", type=int, default=3,
                        help="Use last k memories when summarizing learnings.")
    parser.add_argument("--output-path-prefix", default="save-histories",
                        help="Path prefix to use for saving episode transcripts. Default: %(default)s")
    
    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument("--simplifications-preset", choices=['easy'],
                                      help="Choose a preset among: 'easy' (apply all possible simplifications).")
    simplification_group.add_argument("--teleport", action="store_true",
                                      help="Lets agents instantly move to any location.")
    simplification_group.add_argument("--self-watering-plants", action="store_true",
                                      help="Plants do not have to be frequently watered.")
    simplification_group.add_argument("--open-containers", action="store_true",
                                      help="All containers are opened by default.")
    simplification_group.add_argument("--open-doors", action="store_true",
                                      help="All doors are opened by default.")
    simplification_group.add_argument("--no-electrical", action="store_true",
                                      help="Remove the electrical actions (reduces the size of the action space).")

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    print("ScienceWorld API Examples - STEP Agent")
    # Parse command line arguments
    args = parse_args()
    args["simplification_str"] = build_simplification_str(args)
    stepAgent(args)


if __name__ == "__main__":
    main()