from crewai import Task
from agents import blog_researcher, blog_writer

research_task = Task(
    name="AI Content Research",
    description=(
        "Research {topic} on YouTube. Analyze video content and extract: "
        "- Key concepts\n- Practical examples\n- Technical details"
    ),
    expected_output=(
        "A structured report with:\n"
        "1. Key findings from videos\n"
        "2. Comparison of different approaches\n"
        "3. Recommended resources\n"
        "Format: Markdown with timestamps"
    ),
    agent=blog_researcher,
    async_execution=False
)

write_task = Task(
    name="Blog Post Creation",
    description=(
        "Create a comprehensive blog post about {topic} using the research. "
        "Include:\n- Introduction\n- Technical explanations\n- Examples\n- Conclusion"
    ),
    expected_output=(
        "A well-structured blog post in Markdown format with:\n"
        "1. Engaging title\n2. Introduction\n3. 3-5 main sections\n"
        "4. Code examples (if applicable)\n5. Conclusion\n"
        "6. References to YouTube videos"
    ),
    agent=blog_writer,
    output_file="blog_post.md",
    context=[research_task]
)