"""Example skill showing how to extend Dawu Agent capabilities.

Skills are auto-discovered Python modules that register tools with the agent.
Place your custom skills in this directory.

Example:
    from dawu_agent.tools.registry import tool

    @tool(name="my_custom_tool", description="Does something useful")
    async def my_custom_tool(input_data: str) -> dict:
        return {"result": f"Processed: {input_data}"}
"""
