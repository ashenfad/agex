"""
Geospatial Routing and Visualization

Agent uses osmnx to find routes and folium to create interactive map visualizations.
Demonstrates integration with specialized libraries for geospatial analysis.
"""

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox

from agex import Agent, MemberSpec, Versioned, connect_llm
from agex.helpers import register_stdlib

routy = Agent(
    name="routy",
    primer="You assist the user finding routes (OSMnx) and visualizing them (Folium).",
    llm_client=connect_llm(provider="openai", model="gpt-5"),
    timeout_seconds=60.0,  # agent will need to download openstreetmap data
    max_iterations=20,
)

# give our agent standard lib modules like 'math'
register_stdlib(routy)

# our agent may not be familiar with osmnx, so we'll highlight parts
routy.module(
    ox,
    visibility="low",
    recursive=True,
    configure={
        # elevate for high visibility
        "geocoder.geocode": MemberSpec(visibility="high"),
        "graph.graph_from_bbox": MemberSpec(visibility="high"),
        "distance.nearest_nodes": MemberSpec(visibility="high"),
        "distance.nearest_edges": MemberSpec(visibility="high"),
        "routing.shortest_path": MemberSpec(visibility="high"),
        "routing.k_shortest_paths": MemberSpec(visibility="high"),
        "routing.route_to_gdf": MemberSpec(visibility="high"),
        "convert.graph_to_gdfs": MemberSpec(visibility="high"),
    },
)

# our agent may not be familiar with folium, so we'll highlight parts
routy.module(
    folium,
    visibility="low",
    recursive=True,
    configure={
        # elevate for high visibility
        "Map": MemberSpec(visibility="high"),
        "Marker": MemberSpec(visibility="high"),
        "PolyLine": MemberSpec(visibility="high"),
    },
)

# as osmnx uses nx and geopandas artifacts, our agent may want these as well
routy.module(nx, visibility="low", recursive=True)
routy.module(gpd, visibility="low", recursive=True)


@routy.task
def route(prompt: str) -> folium.Map:  # type: ignore[return-value]
    "Find a route given the prompt and return a Folium map."
    pass


def main():
    state = Versioned()
    from IPython.display import display

    map = route(
        "I'd like drive from Albany, OR to Corvallis, OR", state=state, on_event=display
    )
    map.save("examples/routy-a.html")
    print("Found map from Albany to Corvallis")

    map = route(
        "Hwy 20 is closed between Albany and Corvallis, other options?", state=state
    )
    map.save("examples/routy-b.html")
    print("Found map from Albany to Corvallis (no hwy 20)")


if __name__ == "__main__":
    # Run with: python examples/routy.py OR python -m examples.routy
    main()
