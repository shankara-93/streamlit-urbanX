import os
import json
import logging
from typing import Optional, Tuple, Dict, Union, List, Any
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
# from streamlit_extras.row import row # Removed as potentially unused, re-add if needed
import leafmap.foliumap as leafmap
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, shape
from shapely.validation import make_valid
import folium # Import Folium directly for MarkerCluster
from folium.plugins import MarkerCluster
# import tempfile # No longer needed
from pathlib import Path # Keep if used elsewhere, otherwise removable
import time
import requests # <-- Keep for API calls
# from io import BytesIO # Removed as not needed here

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_CRS = "EPSG:4326"
DEFAULT_CENTER = [52.06, 5.20]
DEFAULT_ZOOM = 15
DEFAULT_BASEMAP = "CartoDB.Positron"
LOGO_PATH = r".\Ref_images\Artix_Print.png" # Use relative path if logo is in the same directory
# ALLOWED_IMAGE_TYPES removed

# --- Configuration ---
FASTAPI_BACKEND_URL = "https://urbanx-960048471565.europe-west4.run.app/detect-from-url" # !!! IMPORTANT: REPLACE THIS !!!

# --- Configuration and Initialization ---

def set_page_config() -> None:
    st.set_page_config(
        page_title="Parking Analysis App",
        page_icon="🅿️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "mailto:info@artixtechnologies.com",
            "About": """
                **Parking Analysis App by Artix Technologies**
                Input a Google Drive image URL to perform parking analysis.
                Contact: info@artixtechnologies.com
            """,
        },
    )

def load_logo(logo_path: str = LOGO_PATH, width: int = 150) -> None:
    try:
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, width=width)
        else:
            st.sidebar.markdown("### Parking Analysis")
            logger.warning(f"Logo image not found at specified path: {logo_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading logo: {str(e)}")
        logger.error(f"Error loading logo: {str(e)}", exc_info=True)

# --- FastAPI Interaction ---

# Keep call_fastapi_from_url as is
@st.cache_data(show_spinner="Sending request to backend...") # Cache based on URL
def call_fastapi_from_url(endpoint_url: str, image_url: str) -> Optional[Dict]:
    """Sends the Google Drive URL to the FastAPI backend."""
    payload = {"image_url": image_url}
    try:
        logger.info(f"Sending POST request to {endpoint_url} with URL: {image_url}")
        response = requests.post(endpoint_url, json=payload, timeout=600) # Increased timeout further
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Received successful response ({response.status_code}) from URL endpoint.")
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out connecting to {endpoint_url}")
        st.error("The request timed out. The backend might be busy or the image is very large.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to {endpoint_url}")
        st.error(f"Could not connect to the backend service at {endpoint_url}. Please ensure it's running.")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from {endpoint_url}: {e.response.status_code} - {e.response.text}")
        st.error(f"Backend returned an error: {e.response.status_code}")
        try: # Try to get detail from FastAPI's JSON error response
            error_detail = e.response.json().get('detail', e.response.text)
            st.error(f"Details: {error_detail}")
        except json.JSONDecodeError:
            st.error(f"Details: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending request to {endpoint_url}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while communicating with the backend: {e}")
        return None
    except json.JSONDecodeError:
         logger.error(f"Failed to decode JSON response from {endpoint_url}")
         st.error("Received an invalid response from the backend.")
         return None

# Remove call_fastapi_from_file function entirely


# --- (Keep Data Processing functions: convert_polygon_geometry_column, parse_api_response) ---
# These functions now process the JSON *returned by FastAPI*

def convert_polygon_geometry_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # (Keep this function as is)
    if 'polygon_geometry' in gdf.columns and not gdf.empty:
        is_dict_col = gdf['polygon_geometry'].apply(lambda x: isinstance(x, dict)).any()
        if is_dict_col:
            try:
                gdf['polygon_geometry'] = gdf['polygon_geometry'].apply(
                    lambda geom: shape(geom) if isinstance(geom, dict) else geom
                )
                gdf['polygon_geometry'] = gdf['polygon_geometry'].apply(
                    lambda geom: make_valid(geom) if geom and not geom.is_valid else geom
                )
            except Exception as e:
                logger.error(f"Error converting/validating 'polygon_geometry': {e}", exc_info=True)
                st.warning(f"Could not convert/validate all 'polygon_geometry' values: {e}")
    return gdf

@st.cache_data(show_spinner="Processing analysis results...")
def parse_api_response(api_response_json: Dict) -> Tuple[Optional[Dict], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[List], Optional[gpd.GeoDataFrame]]:
    """Parses the JSON dictionary received from the FastAPI backend."""
    # (Keep this function as is, assuming FastAPI response structure is consistent)
    logger.info("Parsing API response data received from backend...")
    start_time = time.time()
    if not isinstance(api_response_json, dict) or not api_response_json:
        logger.error("Received invalid or empty JSON structure from FastAPI for parsing")
        st.error("Received invalid or empty data structure from the backend.")
        return None, None, None, None, None

    json_data = api_response_json # Assuming top-level structure matches

    try:
        analysis_metadata = json_data.get("analysis_metadata", {})
        occupied_parking_fc = json_data.get("occupied_parking", {})
        empty_parking_fc = json_data.get("empty_parking", {})
        image_data_dict = json_data.get("image_metadata", {})
        image_bbox = image_data_dict.get("image_bbox") if isinstance(image_data_dict, dict) else None
        image_bbox_centers_fc = image_data_dict.get("image_bbox_centers", {}) if isinstance(image_data_dict, dict) else {}

        gdf_occupied = gpd.GeoDataFrame(geometry=[], crs=DEFAULT_CRS)
        gdf_empty = gpd.GeoDataFrame(geometry=[], crs=DEFAULT_CRS)
        gdf_image_centers = gpd.GeoDataFrame(geometry=[], crs=DEFAULT_CRS)

        if isinstance(occupied_parking_fc, dict) and occupied_parking_fc.get("features"):
            try:
                gdf_occupied = gpd.GeoDataFrame.from_features(occupied_parking_fc["features"], crs=DEFAULT_CRS)
                if not gdf_occupied.empty:
                    gdf_occupied['geometry'] = gdf_occupied['geometry'].apply(
                        lambda geom: make_valid(geom) if geom and not geom.is_valid else geom
                    )
                    gdf_occupied = convert_polygon_geometry_column(gdf_occupied)
                logger.info(f"Processed {len(gdf_occupied)} occupied parking features from backend response.")
            except Exception as e:
                logger.error(f"Error processing occupied parking features from backend response: {e}", exc_info=True)
                st.warning(f"Could not process all occupied parking features from backend: {e}")

        if isinstance(empty_parking_fc, dict) and empty_parking_fc.get("features"):
            try:
                gdf_empty = gpd.GeoDataFrame.from_features(empty_parking_fc["features"], crs=DEFAULT_CRS)
                if not gdf_empty.empty:
                    gdf_empty['geometry'] = gdf_empty['geometry'].apply(
                         lambda geom: make_valid(geom) if geom and not geom.is_valid else geom
                    )
                logger.info(f"Processed {len(gdf_empty)} empty parking features from backend response.")
            except Exception as e:
                logger.error(f"Error processing empty parking features from backend response: {e}", exc_info=True)
                st.warning(f"Could not process all empty parking features from backend: {e}")

        if isinstance(image_bbox_centers_fc, dict) and image_bbox_centers_fc.get("features"):
            try:
                gdf_image_centers = gpd.GeoDataFrame.from_features(image_bbox_centers_fc["features"], crs=DEFAULT_CRS)
                if not gdf_image_centers.empty:
                     gdf_image_centers['geometry'] = gdf_image_centers['geometry'].apply(
                         lambda geom: make_valid(geom) if geom and not geom.is_valid else geom
                     )
                logger.info(f"Processed {len(gdf_image_centers)} image center features from backend response.")
            except Exception as e:
                logger.error(f"Error processing image center features from backend response: {e}", exc_info=True)
                st.warning(f"Could not process image center features from backend: {e}")

        if analysis_metadata.get("error"):
            error_msg = analysis_metadata['error']
            logger.error(f"Analysis error reported in backend metadata: {error_msg}")
            st.error(f"Analysis error from backend: {error_msg}")

        end_time = time.time()
        logger.info(f"Finished parsing backend API response in {end_time - start_time:.2f} seconds.")
        return analysis_metadata, gdf_occupied, gdf_empty, image_bbox, gdf_image_centers

    except Exception as e:
        logger.error(f"Critical error during backend response parsing: {str(e)}", exc_info=True)
        st.error(f"A critical error occurred while processing the results from the backend: {str(e)}")
        return None, None, None, None, None


# --- (Keep Visualization functions: multipoint_to_dataframe, calculate_center, leaf_map_plot) ---
def multipoint_to_dataframe(geo_dataframe: gpd.GeoDataFrame) -> pd.DataFrame:
    # (Keep this function as is)
    locations_parking = []
    if 'geometry' not in geo_dataframe.columns:
        logger.warning("GeoDataFrame provided to multipoint_to_dataframe lacks 'geometry' column.")
        return pd.DataFrame(columns=["longitude", "latitude", "weight"])
    active_geom_col = geo_dataframe.geometry.name
    for geom in geo_dataframe[active_geom_col].dropna():
        if geom.is_empty: continue
        if geom.geom_type == 'MultiPoint':
            num_points = len(geom.geoms)
            for point in geom.geoms:
                if point and not point.is_empty: locations_parking.append([point.x, point.y, num_points])
        elif geom.geom_type == 'Point': locations_parking.append([geom.x, geom.y, 1])
    return pd.DataFrame(locations_parking, columns=["longitude", "latitude", "weight"])

def calculate_center(bounds: List[float]) -> List[float]:
    # (Keep this function as is)
    try:
        if len(bounds) != 4: raise ValueError(f"Expected 4 bounds values, got {len(bounds)}")
        miny, minx, maxy, maxx = map(float, bounds)
        center_y = (miny + maxy) / 2; center_x = (minx + maxx) / 2
        if not (-90 <= center_y <= 90 and -180 <= center_x <= 180): raise ValueError(f"Calculated center [{center_y}, {center_x}] is outside valid lat/lon range.")
        return [center_y, center_x]
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"Invalid bounds format or values: {bounds}. Using default center. Error: {e}")
        return DEFAULT_CENTER

def leaf_map_plot(
    intersecting_data: Optional[gpd.GeoDataFrame],
    non_intersecting_data: Optional[gpd.GeoDataFrame],
    display_bounds: Optional[List[float]],
    all_centroids_gdf: Optional[gpd.GeoDataFrame],
    show_occupied_poly: bool = True, show_empty_poly: bool = True,
    show_parked_cars: bool = True, show_all_cars: bool = True, show_heatmap: bool = True,
) -> Optional[leafmap.Map]:
    # (Keep this function as is)
    logger.info(f"Generating map. Layers(occ_poly={show_occupied_poly}, emp_poly={show_empty_poly}, parked={show_parked_cars}, all={show_all_cars}, heat={show_heatmap})")
    map_start_time = time.time()
    try:
        center_lat_long = calculate_center(display_bounds) if display_bounds else DEFAULT_CENTER
        m = leafmap.Map(center=center_lat_long, zoom=DEFAULT_ZOOM, height="450px", widescreen=False)
        m.add_basemap(DEFAULT_BASEMAP)

        # (Layer logic remains the same)
        # Layer 1: Occupied parking areas (Polygons)
        if show_occupied_poly and intersecting_data is not None and not intersecting_data.empty and 'polygon_geometry' in intersecting_data.columns:
            cols_to_show = ["osmid", "Category", "occupancy", "Capacity", "area_m2"]
            available_cols = [col for col in cols_to_show if col in intersecting_data.columns]
            temp_gdf = intersecting_data[['polygon_geometry'] + available_cols].copy()
            temp_gdf = temp_gdf.set_geometry('polygon_geometry')
            valid_geoms_gdf = temp_gdf[temp_gdf.geometry.notna() & temp_gdf.geometry.is_valid & ~temp_gdf.geometry.is_empty].copy()
            if not valid_geoms_gdf.empty:
                m.add_gdf(valid_geoms_gdf, layer_name="Parking Area (Occupied)", style={'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.4, 'weight': 1}, info_mode='on_click', zoom_to_layer=False)
            else: logger.warning("No valid/non-empty geometries found in 'polygon_geometry' for occupied parking layer.")

        # Layer 2: Empty parking areas (Polygons)
        if show_empty_poly and non_intersecting_data is not None and not non_intersecting_data.empty and 'geometry' in non_intersecting_data.columns:
            cols_to_show_empty = ["osmid", "Category", "Capacity", "area_m2"]
            available_cols_empty = [col for col in cols_to_show_empty if col in non_intersecting_data.columns]
            temp_gdf_empty = non_intersecting_data[['geometry'] + available_cols_empty].copy()
            temp_gdf_empty = temp_gdf_empty.set_geometry('geometry')
            valid_geoms_gdf_empty = temp_gdf_empty[temp_gdf_empty.geometry.notna() & temp_gdf_empty.geometry.is_valid & ~temp_gdf_empty.geometry.is_empty].copy()
            if not valid_geoms_gdf_empty.empty:
                m.add_gdf(valid_geoms_gdf_empty, layer_name="Parking Area (Empty)", style={'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.4, 'weight': 1}, info_mode='on_click', zoom_to_layer=False)
            else: logger.warning("No valid/non-empty geometries found for empty parking layer.")

        # Layer 3: Car centroids in parking (Clustered)
        if show_parked_cars and intersecting_data is not None and not intersecting_data.empty and 'geometry' in intersecting_data.columns:
            points_gdf = intersecting_data.set_geometry('geometry')
            valid_points_gdf = points_gdf[points_gdf.geometry.notna() & points_gdf.geometry.is_valid & ~points_gdf.geometry.is_empty].copy()
            if not valid_points_gdf.empty:
                logger.info(f"Adding {len(valid_points_gdf)} parked car locations with clustering.")
                def get_point_coords(geom):
                    if geom.geom_type == 'Point': return (geom.y, geom.x)
                    if geom.geom_type == 'MultiPoint' and not geom.is_empty: return (geom.geoms[0].y, geom.geoms[0].x)
                    return (None, None)
                locations = valid_points_gdf.geometry.apply(get_point_coords).tolist()
                valid_locations = [(lat, lon) for lat, lon in locations if lat is not None]
                if valid_locations:
                    popup_texts = [f"Parked Car<br>OSMID: {row.get('osmid', 'N/A')}" for _, row in valid_points_gdf.iterrows()]
                    mc = MarkerCluster(name="Cars in Parking (Clustered)", overlay=True, control=True, options={'maxClusterRadius': 30})
                    for i, (lat, lon) in enumerate(valid_locations):
                         folium.CircleMarker(location=[lat, lon], radius=4, popup=popup_texts[i] if i < len(popup_texts) else "Parked Car", color='green', fill=True, fill_color='green', fill_opacity=0.7).add_to(mc)
                    m.add_child(mc)
                else: logger.warning("No valid point coordinates found for parked cars.")

        # Prepare data for All Cars and Heatmap
        df_road = pd.DataFrame(); all_car_locations = []; all_car_popup_texts = []
        if (show_all_cars or show_heatmap) and all_centroids_gdf is not None and not all_centroids_gdf.empty and 'geometry' in all_centroids_gdf.columns:
            all_centroids_temp_gdf = all_centroids_gdf.set_geometry('geometry')
            valid_all_centroids_gdf = all_centroids_temp_gdf[all_centroids_temp_gdf.geometry.notna() & all_centroids_temp_gdf.geometry.is_valid & ~all_centroids_temp_gdf.geometry.is_empty].copy()
            if not valid_all_centroids_gdf.empty:
                logger.info(f"Processing {len(valid_all_centroids_gdf)} total car locations.")
                if show_heatmap: df_road = multipoint_to_dataframe(valid_all_centroids_gdf[['geometry']])
                if show_all_cars:
                    def get_all_point_coords(geom):
                        coords = []
                        if geom.geom_type == 'Point': coords.append((geom.y, geom.x))
                        elif geom.geom_type == 'MultiPoint':
                            for pt in geom.geoms:
                                if pt and not pt.is_empty: coords.append((pt.y, pt.x))
                        return coords
                    all_coords_list = valid_all_centroids_gdf.geometry.apply(get_all_point_coords)
                    for coords in all_coords_list:
                         all_car_locations.extend(coords); all_car_popup_texts.extend(["Detected Car"] * len(coords))

        # Layer 4: All detected car centroids (Clustered)
        if show_all_cars and all_car_locations:
            logger.info(f"Adding {len(all_car_locations)} total car locations with clustering.")
            mc_all = MarkerCluster(name="All Detected Cars (Clustered)", overlay=True, control=True, options={'maxClusterRadius': 40})
            for i, (lat, lon) in enumerate(all_car_locations):
                 folium.CircleMarker(location=[lat, lon], radius=3, popup=all_car_popup_texts[i] if i < len(all_car_popup_texts) else "Detected Car", color='purple', fill=True, fill_color='purple', fill_opacity=0.6).add_to(mc_all)
            m.add_child(mc_all)
        elif show_all_cars: logger.warning("No valid locations found for 'All Detected Cars' layer.")

        # Layer 5: Heatmap
        if show_heatmap and not df_road.empty:
            logger.info("Adding heatmap layer.")
            try: m.add_heatmap(df_road, latitude="latitude", longitude="longitude", value="weight", name="Car Density Heatmap", radius=15, blur=10)
            except Exception as heat_err: logger.error(f"Could not add heatmap layer: {heat_err}", exc_info=True); st.warning("Could not generate heatmap layer.")
        elif show_heatmap: logger.warning("No data available for heatmap generation.")

        # End map generation
        m.add_layer_control()
        map_end_time = time.time()
        logger.info(f"Map generation finished in {map_end_time - map_start_time:.2f} seconds.")
        return m
    except Exception as e:
        logger.error(f"Error generating map: {str(e)}", exc_info=True)
        st.error(f"An error occurred while generating the map: {str(e)}")
        return None


# --- (Keep UI Components: display_metrics, display_map_and_insights) ---
# These should work fine if parse_api_response provides the correct data structure

def display_metrics(analysis_metadata: Dict) -> None:
    """Display the metrics cards based on data from FastAPI."""
    # (Keep this function as is)
    city_name = analysis_metadata.get('city_name', 'N/A')
    num_parking = analysis_metadata.get('num_parking', 'N/A')
    parking_capacity = analysis_metadata.get('parking_capacity', 'N/A')
    total_parking_area = analysis_metadata.get('total_parking_area(sq.m)', 'N/A')

    occupancy_counts = analysis_metadata.get('occupancy_per(counts)', 0)
    occupancy_area = analysis_metadata.get('occupancy_per(area)', 0)
    total_occupied_cars = analysis_metadata.get('Total Occupied (Cars)', 'N/A')
    avg_occupancy_cars = analysis_metadata.get('Avg Occupancy (Cars)', 'N/A')
    parked_area_sqm = analysis_metadata.get('parked_area(sq.m)', 'N/A')
    occupied_parking_areas_count = analysis_metadata.get('Occupied Parking areas', 'N/A')

    with st.container(height=325, border=True): # Adjust height as needed
        col1, col2 = st.columns((0.4, 0.6)) # Adjust column ratios if needed
        with col1.container(border=False):
            st.markdown("""<h5 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Image/Area Metadata</h5>""", unsafe_allow_html=True)
            st.markdown(" ") # Add space
            row1_cols = st.columns(2)
            row1_cols[0].metric(label="Area Name/City", value=city_name) # Make label generic
            row1_cols[1].metric(label="# Parking Spaces Defined", value=num_parking)
            row2_cols = st.columns(2)
            row2_cols[0].metric(label="Est. Total Capacity", value=parking_capacity)
            row2_cols[1].metric(label="Total Parking Area (m²)", value=f"{total_parking_area:.1f}" if isinstance(total_parking_area, (int, float)) else "N/A" )

        with col2.container(border=False):
            st.markdown("""<h5 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Parking Detection Results</h5>""", unsafe_allow_html=True)
            st.markdown(" ") # Add space
            met_row1_cols = st.columns(2)
            met_row1_cols[0].metric(label="Occupancy % (by count)", value=f"{occupancy_counts:.1f}%" if isinstance(occupancy_counts, (int, float)) else "N/A")
            met_row1_cols[1].metric(label="Occupancy % (by area)", value=f"{occupancy_area:.1f}%" if isinstance(occupancy_area, (int, float)) else "N/A")

            met_row2_cols = st.columns(4)
            met_row2_cols[0].metric(label="# Occupied Spaces", value=total_occupied_cars)
            met_row2_cols[1].metric(label="Avg Occ/Area", value=f"{avg_occupancy_cars:.1f}" if isinstance(avg_occupancy_cars, (int, float)) else "N/A") # Abbreviate label
            met_row2_cols[2].metric(label="Parked Area (m²)", value=f"{parked_area_sqm:.1f}" if isinstance(parked_area_sqm, (int, float)) else "N/A")
            met_row2_cols[3].metric(label="# Occupied Areas", value=occupied_parking_areas_count) # Abbreviate label

            try:
                 style_metric_cards(border_radius_px=10)
            except NameError:
                 logger.warning("style_metric_cards not available. Install streamlit_extras.")


def display_map_and_insights(
    gdf_occupied: Optional[gpd.GeoDataFrame],
    gdf_empty: Optional[gpd.GeoDataFrame],
    image_bbox: Optional[List[float]],
    gdf_image_centers: Optional[gpd.GeoDataFrame],
    map_options: Dict[str, Any]
) -> None:
    """Display the map and data insights side by side."""
    # (Keep this function as is)
    with st.container(border=True):
        map_col, insights_col = st.columns((0.6, 0.4))
        with map_col:
            st.markdown("""<h4 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Occupancy City Map</h4>""", unsafe_allow_html=True)
            st.markdown(" ")
            with st.spinner("Generating map..."):
                 the_map = leaf_map_plot(gdf_occupied, gdf_empty, image_bbox, gdf_image_centers, **map_options)
            if the_map: the_map.to_streamlit(use_container_width=True)
            else: st.warning("Map could not be generated.")
        with insights_col:
            st.markdown("""<h4 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Occupied Parking Insights</h4>""", unsafe_allow_html=True)
            st.markdown(" ")
            if gdf_occupied is not None and not gdf_occupied.empty:
                columns_to_show = ["osmid", "Category", "occupancy", "Capacity", "area_m2", "occupancy(%)"]
                available_columns = [col for col in columns_to_show if col in gdf_occupied.columns]
                if available_columns:
                    table_df = gdf_occupied[available_columns].copy()
                    for col in table_df.select_dtypes(include=['float']).columns: table_df[col] = table_df[col].round(2)
                    table_df = table_df.drop(columns=[col for col in ['geometry', 'polygon_geometry'] if col in table_df.columns], errors='ignore')
                    column_config_dict = {}
                    for col_name in available_columns:
                        capitalized_label = col_name.replace('_', ' ').capitalize()
                        if col_name == "osmid": column_config_dict[col_name] = st.column_config.TextColumn("OSM ID", help="OpenStreetMap ID")
                        elif col_name == "area_m2": column_config_dict[col_name] = st.column_config.NumberColumn("Area (sq.m)", format="%.1f")
                        elif col_name == "occupancy(%)": column_config_dict[col_name] = st.column_config.NumberColumn("Occupancy (%)", format="%.1f%%")
                        elif col_name == "occupancy": column_config_dict[col_name] = st.column_config.NumberColumn("Occupancy (Count)")
                        elif isinstance(table_df[col_name].dtype, (int, float)): column_config_dict[col_name] = st.column_config.NumberColumn(capitalized_label)
                        else: column_config_dict[col_name] = st.column_config.Column(label=capitalized_label)
                    st.dataframe(table_df, use_container_width=True, height=600, hide_index=True, column_config=column_config_dict)
                else: st.info("No relevant data columns found in occupied parking data.")
            else: st.info("No occupied parking data available for insights.")

# --- Sidebar Definition (Updated) ---

def display_sidebar_options() -> Tuple[str, bool, Dict[str, Any], st.empty]:
    """Manages sidebar input widgets and map options."""
    st.sidebar.header("Input Image Source")

    # --- Status Placeholder ---
    status_placeholder = st.sidebar.empty()
    status_placeholder.info("ℹ️ Provide a Google Drive URL, then click 'Process'.")

    # --- Input Method: Google Drive URL ---
    google_drive_url = st.sidebar.text_input(
        "Google Drive Image URL:",
        help="Paste the shareable link to an image file on Google Drive.",
        key="gdrive_url_input" # Add a key for potential state management
    )
    # --- Process Button (Moved here) ---
    process_button_clicked = st.sidebar.button(
        "Process Image",
        key="process_button",
        type="primary",
        use_container_width=True
    )

    st.sidebar.divider() # Divider after the main input section

    # --- Map Display Options ---
    st.sidebar.subheader("Map Layer Options")
    show_occupied_poly = st.sidebar.toggle("Show Occupied Areas", value=True, help="Display red polygons for occupied parking areas.")
    show_empty_poly = st.sidebar.toggle("Show Empty Areas", value=True, help="Display blue polygons for empty parking areas.")
    show_parked_cars = st.sidebar.toggle("Show Parked Cars", value=True, help="Display clustered green markers for cars within parking areas.")
    show_all_cars = st.sidebar.toggle("Show All Detected Cars", value=False, help="Display clustered purple markers for *all* detected cars (can be slow).")
    show_heatmap = st.sidebar.toggle("Show Density Heatmap", value=True, help="Display a heatmap of car density.")

    st.sidebar.divider()

    map_options = {
        "show_occupied_poly": show_occupied_poly,
        "show_empty_poly": show_empty_poly,
        "show_parked_cars": show_parked_cars,
        "show_all_cars": show_all_cars,
        "show_heatmap": show_heatmap,
    }

    # Return the URL, button state, map options, and placeholder
    return google_drive_url, process_button_clicked, map_options, status_placeholder


# --- Main Application (Updated) ---

def main() -> None:
    """Main application function."""
    set_page_config()
    load_logo()

    # --- Get Inputs and Button State from Sidebar ---
    google_drive_url, process_button_clicked, map_options, status_placeholder = display_sidebar_options()

    # --- Logic Triggered by Button Click ---
    if process_button_clicked:

        # Input Validation & Backend Call
        api_response_json = None
        data_source_name = None

        if google_drive_url: # Check if URL is provided
            data_source_name = f"Google Drive URL: ...{google_drive_url[-30:]}" # Shorten display name
            status_placeholder.info(f"Processing: {data_source_name}")
            logger.info(f"Processing started for URL: {google_drive_url}")

            # Define the correct endpoint for URL processing
            api_response_json = call_fastapi_from_url(FASTAPI_BACKEND_URL, google_drive_url)

        else:
            # No URL provided when button was clicked
            st.sidebar.warning("Please provide a Google Drive URL first.")
            status_placeholder.warning("⚠️ No Google Drive URL provided.")
            logger.warning("Process button clicked but no URL provided.")


        # --- Process and Display Data if Backend Call Successful ---
        if api_response_json:
            status_placeholder.success(f"✓ Processing complete for {data_source_name}")
            logger.info(f"Successfully received response for {data_source_name}. Parsing results.")

            # Parse data (cached based on input URL)
            analysis_metadata, gdf_occupied, gdf_empty, image_bbox, gdf_image_centers = parse_api_response(api_response_json)

            # Check if parsing was successful before displaying
            if analysis_metadata is not None: # Check if parsing returned data
                 # Display results in the main area
                 st.header("Analysis Results")
                 display_metrics(analysis_metadata)
                 display_map_and_insights(
                     gdf_occupied, gdf_empty, image_bbox, gdf_image_centers, map_options
                 )
                 logger.info(f"Application displayed results successfully for {data_source_name}.")
            else:
                # Error during parsing (already logged/shown by parse_api_response)
                status_placeholder.error(f"❌ Failed to process results for {data_source_name}")
                logger.error(f"Application halted due to data processing/parsing error from backend for {data_source_name}.")

        elif google_drive_url: # Only show error if an attempt was made (URL was provided but backend call failed)
             status_placeholder.error(f"❌ Failed to get results for {data_source_name}")
             # Error messages are shown by the call_fastapi_from_url function

    # --- Add Contact Info LAST in the sidebar ---
    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        **📍 Address:** Oder 20, UNIT A1164, The Hague, 2491DC\n
        **📧 Email:** info@artixtechnologies.com\n
        **📞 Phone:** +31 0622385211\n
        **🏢 KVK No:** 94868468
        """
    )

if __name__ == "__main__":
    #if FASTAPI_BACKEND_URL == "YOUR_FASTAPI_CLOUD_RUN_URL_HERE":
        #st.error("🚨 Critical Error: FastAPI backend URL is not configured. Please set `FASTAPI_BACKEND_URL` in the script.")
        #st.stop()
    main()