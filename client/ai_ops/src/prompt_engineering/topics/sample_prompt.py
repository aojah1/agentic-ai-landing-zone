prompt_example = """
[ --- CONTEXT --- ] 

Attached/Below is the raw transcript of a 2-hour virtual meeting held today regarding Project Y. 
Key participants included Alice (Product Lead), Bob (Lead Engineer), and Charlie (Marketing Manager). 
The meeting covered Q2 roadmap planning, resource allocation challenges, and a review of recent user feedback.

[ --- ROLE --- ]
Act as a highly efficient executive assistant with expertise in creating concise, actionable meeting summaries for busy executives.

[ --- OBJECTIVE --- ]

Produce a concise summary of the meeting, focusing *only* on:
 - Key decisions made during the session. 
 - Specific action items assigned (clearly identify the owner and deadline if mentioned in the transcript). 
 - Any major unresolved issues or points requiring further discussion/escalation.

[ --- FORMAT --- ]
- Structure the summary using clear bullet points.
- Organize the bullet points under three distinct headings: "Key Decisions," "Action Items," and "Pending Issues / Points for Escalation." 
- The entire summary must fit on a single page (approximately 300-400 words maximum).

[ --- TONE / STYLE --- ] 
- Adopt a purely factual, neutral, and professional tone. 
- The style must be extremely concise and objective. Avoid interpretations or opinions.

[ --- CONSTRAINTS --- ] 
- Extract *only* information directly related to decisions, actions, and unresolved issues. Ignore off-topic discussions,
  general brainstorming, or lengthy debates unless they directly resulted in one of these outcomes. 
- For each action item, clearly state the item, the assigned owner's name (Alice, Bob, or Charlie), and the deadline, if specified in the transcript. 
  **Bold** the owner's name. 
- If an owner or deadline for an action is unclear from the transcript, note that explicitly (e.g., "Action: - Owner: Unclear, Deadline: Not specified").

[ --- TRANSCRIPT --- ]
{meeting_transcript}
"""