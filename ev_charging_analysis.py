import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import folium
from sklearn.cluster import KMeans
from shapely import wkt

# === Load EV station data ===
ev_stations = pd.read_csv("ev_stations.csv")

# Create geometry from Latitude/Longitude
ev_stations["geometry"] = ev_stations.apply(
    lambda row: Point(row["Longitude"], row["Latitude"]) if pd.notnull(row["Latitude"]) and pd.notnull(row["Longitude"]) else None,
    axis=1
)
gdf_stations = gpd.GeoDataFrame(ev_stations, geometry="geometry", crs="EPSG:4326")

# === Load EV usage data ===
usage_data = pd.read_csv("usage_data.csv")

usage_data["geometry"] = usage_data["Vehicle Location"].apply(
    lambda x: wkt.loads(x) if pd.notnull(x) and "POINT" in x else None
)
gdf_usage = gpd.GeoDataFrame(usage_data, geometry="geometry", crs="EPSG:4326")

# === Exploratory Analysis ===
# Plot top 20 cities by station count
gdf_stations["region"] = gdf_stations["City"].astype(str) + ", " + gdf_stations["State"].astype(str)
top_regions = gdf_stations["region"].value_counts().head(20)
top_regions.plot(kind="barh", figsize=(8, 6), title="Top 20 Regions by Station Count")
plt.tight_layout()
plt.savefig("top_station_regions.png")

# Connector type distribution
plt.figure()
sns.countplot(data=ev_stations, x="EV Connector Types")
plt.title("Connector Type Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("connector_distribution.png")

# === Mock Gap Analysis by County ===
gap_df = gdf_usage.groupby("County").agg({
    "VIN (1-10)": "count"
}).rename(columns={"VIN (1-10)": "EV_Count"})

gap_df["Population"] = 100000
gap_df["Median_Income"] = 50000
gap_df["Station_Count"] = gdf_stations.groupby("State").size()

scaler = MinMaxScaler()
gap_df[["EV_Count_Norm", "Pop_Norm"]] = scaler.fit_transform(gap_df[["EV_Count", "Population"]])
gap_df["Gap"] = gap_df["EV_Count_Norm"] - gap_df["Station_Count"] / gap_df["Pop_Norm"]
gap_df.to_csv("gap_analysis.csv")

# === Optional: Dummy Regression Model ===
features = pd.DataFrame({
    "lat": gdf_stations.geometry.y,
    "lon": gdf_stations.geometry.x,
    "connector_type": ev_stations["EV Connector Types"].astype("category").cat.codes
}).dropna()

target = pd.Series([10] * len(features))  # Placeholder target
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Dummy RMSE:", mean_squared_error(y_test, preds, squared=False))
print("Dummy R²:", r2_score(y_test, preds))

# === Clustering for Expansion ===
coords = gdf_usage.dropna(subset=["geometry"])
coords_list = coords["geometry"].apply(lambda pt: [pt.x, pt.y]).tolist()
kmeans = KMeans(n_clusters=5, random_state=0).fit(coords_list)
coords["cluster"] = kmeans.labels_

# === Folium Map ===
m = folium.Map(location=[33.75, -84.4], zoom_start=6)
for _, row in gdf_stations.dropna(subset=["geometry"]).iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color="blue",
        popup=row.get("Station Name", "EV Station")
    ).add_to(m)
m.save("ev_station_map.html")
