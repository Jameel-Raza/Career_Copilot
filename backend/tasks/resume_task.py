from crewai import Task

def resume_task_to_agent(agent, data):

    return Task(
        description=f"""
        You are an expert AI resume writer.

        Generate a professional resume in valid HTML based on the user's inputs:

        - Name: {data.name}
        - Skills: {', '.join(data.skills)}
        - Experience: {data.experience}
        - Template style: {data.template}

        Use clean HTML tags (<div>, <h2>, <ul>, <li>, <p>) and inline CSS if needed.
        Avoid markdown or placeholder text. Make it visually clear and job-market ready.
        """,
        agent=agent,
        expected_output="Complete resume in HTML format, styled and formatted correctly."
    )
