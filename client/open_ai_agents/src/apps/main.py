import asyncio

from src.agent_teams.research_bot.manager import ResearchManager


async def main() -> None:
    query = input("What would you like to research? ")
    await ResearchManager().run(query)


if __name__ == "__main__":
    asyncio.run(main())