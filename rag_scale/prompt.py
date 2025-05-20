rag_prompt = "Context:{context}\n Question:{question}"
eval_prompt = '''You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
Given a user question and the retrieved context passages, assess the overall quality based on the following criteria:

Relevance: How relevant is the retrieved context to the user’s question?

Completeness: Does the retrieved context provide sufficient information to fully or mostly answer the question?

Accuracy: Is the information in the context factually correct and reliable?

Clarity: Is the context clear and easy to understand?

Redundancy: Is there unnecessary or repetitive information in the context?

Please score each criterion on a scale of 1–5 (1 = Very Poor, 5 = Excellent) and write a brief explanation for each score.

Finally, give an overall quality rating (1–5) and briefly summarize your judgment.
Example Input:

{content}
'''
def eval_prompt(content):
    return [
        {"role": "system", "content": "You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. Given a user question and the retrieved context passages, assess the overall quality"},
        {"role": "user", "content": content}
    ]
complete_query_from_context = """You are given an context passage retrieved by a RAG system.
Your goal is to infer the most likely original user question that would lead to this context being relevant.
retrieved context:
{context}
The original question might be:
"""
complete_query_from_ans = """Given intermediate answer containing the facts about the original question, which is unknown, your task is to infer what the orginal question might have been.
Output the most likely original question directly and nothing else.

Here are some examples:

Example 1:
Intermediate answer:
Muhammad Ali was 74 years old when he died.
Alan Turing* was 41 years old when he died.
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?

Intermediate answer:
{answer}
The original question might be:
"""
# Example 2:
# Intermediate answer:
# Craigslist was founded by Craig Newmark.
# Craig Newmark was born on December 6, 1952.
# The original question might be:
# When was the founder of craigslist born?

complete_query_from_subquery = """Given sub-question derived from the original question, which is unknown, your task is to infer what the original question might have been.
Output the most likely original question directly and nothing else.

Here are some examples:

Example 1:
Sub-question:
How old was Muhammad Ali when he died?
How old was Alan Turing when he died?
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?

Sub-question:
{query}
The original question might be:
"""
