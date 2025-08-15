from crewai import Agent
from tools import youtube_channel_tool
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Create a senior blog researcher

blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal="Get the relevant video content for the topic{topic} from Youtube Channels",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, Machine Learning, GenAI and providing suggestions"
    ),
    tools=[youtube_channel_tool],
    allow_delegation=True,
    name="Senior Blog Researcher",
    description="A senior researcher specialized in blog content.",
    skills=["research", "writing", "SEO"],
    experience_level="senior",
)

# Creating a senior blog writer agent with Youtube tool

blog_writer = Agent(
    role = 'Blog Writer',
    goal="Narrate compelling tech stories about the video {topic} from Youtube channels",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narrative that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[youtube_channel_tool],
    allow_delegation=False,
    name="Senior Blog Writer",
    description="A senior writer specialized in blog content.",
    skills=["writing", "SEO", "content creation"],
    experience_level="senior"
)
