# Headless backend for GitHub Actions cloud runner (no GUI)
import matplotlib
matplotlib.use("Agg")

import os
import math
import pytz
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import PchipInterpolator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import Point
from shapely.ops import unary_union

try:
    from shapely.validation import make_valid
except ImportError:
    try:
        from shapely import make_valid
    except ImportError:
        make_valid = lambda g: g

# ==========================================
# 1. CONSTANTS & STORM DATA CONFIGURATION
# ==========================================
E1=120
E2=160
N1=3
N2=35
MACAU_LAT = 22.1595
MACAU_LON = 113.5685
KM_PER_DEG = 111.32

#15, 50, 100, 170, 255, 345, 465
STORM_DATA = {
    "forecast": {
        "hours": np.array([0, 12, 24, 48, 72, 96]),
        "lats": np.array([16.1,17.0,18.8,26.4,33.7,39.2]),
        "lons": np.array([140.5,138.1,135.4,134.1,137.2,154.3]),
        "radii_km": [15, 60, 100, 170, 255]
    },

    #"#DDDFE2","#6DD8FA","#9DD79C","#FFD363","#F78A31","#FF6F6F","#DE82FF"
    "past_track": {
        "lats": np.array([14.2,14.5,14.7,14.8,14.8,14.9,15.1]),
        "lons": np.array([147.8,146.5,145.6,144.9,143.9,142.8,141.9]),
        "colors": ["#6DD8FA","#6DD8FA","#6DD8FA","#6DD8FA","#DDDFE2","#DDDFE2","#DDDFE2"
    },
    "icons": [
        os.path.join('TC logo', 'TD.png'), 
        os.path.join('TC logo', '12H.png'),  
        os.path.join('TC logo', 'TS.png'), 
        os.path.join('TC logo', 'STS.png'),  
        os.path.join('TC logo', 'TS.png'),  
        os.path.join('TC logo', 'Ex.png'), 
        os.path.join('TC logo', 'Ex.png')   
    ],

    "wind_radii": {
        # Format: (start_angle, end_angle, radius_in_km)
        "strong": [(0, 90, 0), 
                   (90, 180, 0), 
                   (180, 270, 0), 
                   (270, 360, 0)],
        "storm": [(0, 90, 0), 
                  (90, 180, 0), 
                  (180, 270, 0), 
                  (270, 360, 0)]
    }
}

# ==========================================
# 2. MATH & GEOMETRY HELPER FUNCTIONS
# ==========================================

def create_smooth_track(hours, lons, lats, points_per_segment=20):
    """Create smooth track using interpolation"""
    if len(hours) < 2:
        return lons, lats, hours
    
    interp_hours = []
    for i in range(len(hours)-1):
        segment = np.linspace(hours[i], hours[i+1], num=points_per_segment)[:-1]
        interp_hours.extend(segment)
    interp_hours.append(hours[-1])
    interp_hours = np.array(interp_hours)
    
    smooth_lons = PchipInterpolator(hours, lons)(interp_hours)
    smooth_lats = PchipInterpolator(hours, lats)(interp_hours)
    
    return smooth_lons, smooth_lats, interp_hours

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def vectorized_haversine(lat1, lon1, lats2, lons2):
    """Vectorized version for calculating distances to arrays of coordinates."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats2)
    delta_phi = np.radians(lats2 - lat1)
    delta_lambda = np.radians(lons2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360

def bearing_to_compass(bearing_deg):
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE","S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = int((bearing_deg + 11.25) % 360 / 22.5)
    return directions[index]

# ==========================================
# 3. PLOTTING HELPER FUNCTIONS
# ==========================================

def set_map_extent(ax, distance_to_macau):
    """Sets the dynamic zoom of the map based on the storm's distance."""
    # Fixed the logic bug here (changed the second < 400 to < 800)
    if distance_to_macau < 200:
        radius_km, lat_offset, lon_offset = 250, 1.2, 0
    elif distance_to_macau < 400:
        radius_km, lat_offset, lon_offset = 500, 2.0, 0
    elif distance_to_macau < 800:
        radius_km, lat_offset = 850, 5.5
        # Custom bounds for the medium zoom
        lat_min = MACAU_LAT - 5.5 - radius_km / KM_PER_DEG
        lat_max = MACAU_LAT + 3.5 + radius_km / KM_PER_DEG
        lon_min = MACAU_LON - 3 - radius_km / (KM_PER_DEG * abs(math.cos(math.radians(MACAU_LAT))))
        lon_max = MACAU_LON + 3 + radius_km / (KM_PER_DEG * abs(math.cos(math.radians(MACAU_LAT))))
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_aspect('equal', adjustable='box')
        return
    else:
        ax.set_extent([E1,E2,N1,N2], crs=ccrs.PlateCarree())
        ax.set_aspect('equal', adjustable='box')
        return

    # Standard bounds for close/mid zoom
    lat_min = MACAU_LAT - lat_offset - radius_km / KM_PER_DEG
    lat_max = MACAU_LAT + radius_km / KM_PER_DEG
    lon_min = MACAU_LON - radius_km / (KM_PER_DEG * abs(math.cos(math.radians(MACAU_LAT))))
    lon_max = MACAU_LON + radius_km / (KM_PER_DEG * abs(math.cos(math.radians(MACAU_LAT))))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_aspect('equal', adjustable='box')

def plot_wind_radii(ax, center_lon, center_lat, quadrants, color, alpha=0.05, linewidth=0.5):
    """Plots the wind radii quadrants."""
    arc_points = []
    for start_angle, end_angle, radius_km in quadrants:
        radius_deg = radius_km / KM_PER_DEG
        theta = np.linspace(start_angle, end_angle, 50)
        x = center_lon + radius_deg * np.cos(np.deg2rad(theta))
        y = center_lat + radius_deg * np.sin(np.deg2rad(theta))
        arc_points.extend(list(zip(x, y)))
        
    arc_points.append(arc_points[0]) # Close the loop
    arc_lons, arc_lats = zip(*arc_points)
    
    ax.plot(arc_lons, arc_lats, color=color, linewidth=linewidth)
    ax.fill(arc_lons, arc_lats, color=color, alpha=alpha)

# ==========================================
# Motion Calculator (Moved outside main function, fixed indent)
# ==========================================
def calc_motion(idx_start, idx_end, label, hours, lats, lons):
    if len(lats) > idx_end:
        lat1, lon1 = lats[idx_start], lons[idx_start]
        lat2, lon2 = lats[idx_end], lons[idx_end]

        dist = haversine(lat1, lon1, lat2, lon2)  # km
        hours_diff = hours[idx_end] - hours[idx_start]
        speed = dist / hours_diff  # km/h

        dir_deg = bearing(lat1, lon1, lat2, lon2)
        dir_compass = bearing_to_compass(dir_deg)

        print(f"{label} Motion: {dir_compass} {speed:.1f} km/h ({dist:.0f} km over {hours_diff}h)")
    else:
        print(f"{label} Motion: Not enough data")

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================
def create_typhoon_map():
    # Load data from config
    hours = STORM_DATA["forecast"]["hours"]
    lons = STORM_DATA["forecast"]["lons"]
    lats = STORM_DATA["forecast"]["lats"]
    radii_deg = np.array(STORM_DATA["forecast"]["radii_km"]) / KM_PER_DEG
    
    center_lat_wind = lats[0]
    center_lon_wind = lons[0]

    # Process smooth tracks and envelopes
    smooth_lons, smooth_lats, interp_hours = create_smooth_track(hours, lons, lats)
    smooth_radii = PchipInterpolator(hours, radii_deg)(interp_hours)

    circles_segment1 = [Point(smooth_lons[i], smooth_lats[i]).buffer(smooth_radii[i]) for i in range(len(smooth_lons)) if interp_hours[i] <= 72]
    circles_segment2 = [Point(smooth_lons[i], smooth_lats[i]).buffer(smooth_radii[i]) for i in range(len(smooth_lons)) if interp_hours[i] > 72]

    envelope_first = unary_union(circles_segment1) if circles_segment1 else None
    envelope_second = unary_union(circles_segment2) if circles_segment2 else None

    # Handle Envelope Overlaps
    idx_72h = int(np.argmin(np.abs(interp_hours - 72)))
    circle_72h_geom = Point(smooth_lons[idx_72h], smooth_lats[idx_72h]).buffer(smooth_radii[idx_72h])
    
    if envelope_second and not envelope_second.is_empty:
        envelope_second = make_valid(envelope_second.difference(circle_72h_geom))
    if envelope_first and not envelope_first.is_empty and envelope_second and not envelope_second.is_empty:
        envelope_second_no_overlap = make_valid(envelope_second.difference(envelope_first))
    else:
        envelope_second_no_overlap = envelope_second

    # Current Location & Math calculations
    distance_to_macau = haversine(MACAU_LAT, MACAU_LON, center_lat_wind, center_lon_wind)
    bearing_deg = bearing(MACAU_LAT, MACAU_LON, center_lat_wind, center_lon_wind)
    direction = bearing_to_compass(bearing_deg)

    # Setup Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    set_map_extent(ax, distance_to_macau)

    # Map Background Features
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.25)
    ax.add_feature(cfeature.LAND, edgecolor="#959a9f", facecolor="#2d363f")
    ax.add_feature(cfeature.OCEAN, facecolor="#222a35")

    # Plot Envelopes
    if envelope_first and not envelope_first.is_empty:
        ax.add_geometries([envelope_first], crs=ccrs.PlateCarree(), facecolor='white', alpha=0.20)
    if envelope_second_no_overlap and not envelope_second_no_overlap.is_empty:
        ax.add_geometries([envelope_second_no_overlap], crs=ccrs.PlateCarree(), facecolor='white', alpha=0.10)

    # Plot Reference circles for forecast points (24h, 48h,72h)
    def add_circle_if_available(idx, radius_km):
        if len(lons) > idx and len(lats) > idx:
            radius_deg = radius_km / KM_PER_DEG
            ax.add_patch(
                plt.Circle(
                    (lons[idx], lats[idx]),
                    radius_deg,
                    color="white",
                    fill=False,
                    linewidth=0.2,
                    linestyle="-",
                    alpha=0.2,
                    transform=ccrs.PlateCarree(),
                )
            )
    add_circle_if_available(2, 100)
    add_circle_if_available(3, 170)
    add_circle_if_available(4, 255)

    # Reference circles for Macao
    labels = ['100 km', '200 km', '400 km', '800 km']
    for km, label in zip([100, 205, 410, 850], labels):
        radius_deg = km / KM_PER_DEG
        circle = plt.Circle((MACAU_LON, MACAU_LAT), radius_deg, color="#949494", fill=False, linewidth=0.5, alpha=0.5, transform=ccrs.PlateCarree(), linestyle='--')
        ax.add_patch(circle)
        ax.text(MACAU_LON, MACAU_LAT - radius_deg - 0.1, label, color='white', alpha=0.5, fontsize=6, ha='center', va='top', transform=ccrs.PlateCarree())
    
    ax.plot(MACAU_LON, MACAU_LAT, 'o', color='white', markersize=5)

    # Plot Forecast Track
    ax.plot(smooth_lons, smooth_lats, color='white', linewidth=1, linestyle='--',zorder = -1)

    # Plot Icons (safe load, skip missing files to avoid crash)
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        icon_path = STORM_DATA["icons"][i]
        if os.path.exists(icon_path):
            img = mpimg.imread(icon_path)
            imagebox = OffsetImage(img, zoom=0.0042)
            ab = AnnotationBbox(imagebox, (lon, lat), frameon=False, transform=ccrs.PlateCarree(), zorder=5)
            ax.add_artist(ab)

    # Plot Past track
    plats = STORM_DATA["past_track"]["lats"]
    plons = STORM_DATA["past_track"]["lons"]
    pt_past = np.arange(len(plons)) 
    t_psmooth = np.linspace(pt_past[0], pt_past[-1], 200)
    
    plon_smooth = PchipInterpolator(pt_past, plons)(t_psmooth)
    plat_smooth = PchipInterpolator(pt_past, plats)(t_psmooth)
    
    dot_colors = STORM_DATA["past_track"]["colors"]
    for i, (plon, plat) in enumerate(zip(plons, plats)):
        ax.plot(plon, plat, marker='o', color=dot_colors[i], markersize=4, zorder=2)
    ax.plot(plons, plats, color='white', linewidth=1, zorder=1)

    # Plot Wind Radii
    plot_wind_radii(ax, center_lon_wind, center_lat_wind, STORM_DATA["wind_radii"]["strong"], 'yellow', alpha=0.05, linewidth=0.3)
    plot_wind_radii(ax, center_lon_wind, center_lat_wind, STORM_DATA["wind_radii"]["storm"], 'red', alpha=0.05, linewidth=0.5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--', zorder=-4)
    gl.top_labels = False
    gl.right_labels = False

    # Timestamp text
    local_tz = pytz.timezone("Asia/Macau")
    now = datetime.now(local_tz)
    minute = (now.minute // 10) * 10
    if now.minute % 10 >= 5:
        minute += 10

    if minute == 60:
        if now.hour == 23:
            now = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            now = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
    else:
       now = now.replace(minute=minute, second=0, microsecond=0)

    date_time = now.strftime("%Y-%m-%d  %H:%M") + " MST"
    ax.text(0.05, 0.95, date_time, transform=ax.transAxes, ha='left', va='top',
        fontsize=10, color='white', alpha=0.9,
        bbox=dict(facecolor='black', alpha=0.3, pad=3, edgecolor='none'), zorder=10)

    # Print Terminal Data / Calculate CPA (Using Vectorized Haversine now)
    print(f"Nowcasting: {center_lat_wind:.1f}°N, {center_lon_wind:.1f}°E")
    print(f"Distance from Macao: {direction} {distance_to_macau:.0f} km")

    distances = vectorized_haversine(MACAU_LAT, MACAU_LON, smooth_lats, smooth_lons)
    closest_index = np.argmin(distances)
    closest_lon, closest_lat = smooth_lons[closest_index], smooth_lats[closest_index]
    closest_distance = distances[closest_index]
    closest_bearing_deg = bearing(MACAU_LAT, MACAU_LON, closest_lat, closest_lon)
    closest_direction = bearing_to_compass(closest_bearing_deg)

    past_distances = vectorized_haversine(MACAU_LAT, MACAU_LON, plat_smooth, plon_smooth)
    past_closest_index = np.argmin(past_distances)
    closest_plon, closest_plat = plon_smooth[past_closest_index], plat_smooth[past_closest_index]
    past_closest_distance = past_distances[past_closest_index]
    past_bearing_deg = bearing(MACAU_LAT, MACAU_LON, closest_plat, closest_plon)
    past_direction = bearing_to_compass(past_bearing_deg)

    print("\nClosest Point of Approach:")
    if past_closest_distance < closest_distance:
        print(f"Past track CPA: {closest_plat:.2f}°N, {closest_plon:.2f}°E ({past_direction} {past_closest_distance:.0f} km)\n")
    else:
        print(f"Forecast track CPA: {closest_lat:.2f}°N, {closest_lon:.2f}°E ({closest_direction} {closest_distance:.0f} km)\n")

    # Call motion calculator
    calc_motion(0, 1, "12H Avg", hours, lats, lons)
    calc_motion(0, 2, "24H Avg", hours, lats, lons)

    # Save and Show (plt.show() does nothing on headless server)
    plt.savefig('2608.png', dpi=1300, bbox_inches='tight')
    plt.close()
    plt.show()

if __name__ == "__main__":
    create_typhoon_map()
