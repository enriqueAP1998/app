import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from numba import njit, prange
import random
import geopandas as gpd
import utm
import pandas as pd
from tqdm import tqdm

# -------------------------------
# Random Demand Generation
# -------------------------------
@njit()
def generate_random_demand(num_passengers, time_limit):
    """
    Generate random passenger arrival times from a homogeneous distribution.

    Args:
        num_passengers (int): Number of passengers.
        time_limit (int): Upper limit of the time interval (exclusive).

    Returns:
        list: Sorted list of passenger arrival times.
    """
    # Generate random integers between 0 and time_limit - 1
    arrival_times = [random.randint(0, time_limit - 1) for _ in range(num_passengers)]
    arrival_times.sort()
    return arrival_times


# -------------------------------
# Failure Rate Computation
# -------------------------------
@njit()
def generate_random_demand(num_passengers, time_limit):
    return np.random.randint(0, time_limit, num_passengers)

@njit(fastmath = True, parallel = True)
def compute_failure_rate(population_area, num_platforms, num_parkings, N_simulations, 
                         num_vehicles_initial, fraction_of_passengers, taxi_time,
                         max_wait_platform, max_wait_parking, max_wait_departure):
    """
    Compute the failure rate of a vertiport simulation considering maximum acceptable waiting times.

    Args:
        population_area (int): Population within the area.
        num_platforms (int): Number of landing platforms.
        num_parkings (int): Number of parking spots.
        N_simulations (int): Number of simulations to run.
        num_vehicles_initial (int): Initial number of vehicles at the vertiport.
        fraction_of_passengers (float): Fraction of population using the vertiport.
        taxi_time (int): Minimum time required between platform uses.
        max_wait_platform (int): Maximum acceptable waiting time for platform access (minutes).
        max_wait_parking (int): Maximum acceptable waiting time for parking spots (minutes).
        max_wait_departure (int): Maximum acceptable waiting time for departure vehicle availability (minutes).

    Returns:
        float: Failure rate (ratio of simulations exceeding waiting times).
    """
    num_passengers = int(population_area * fraction_of_passengers)
    num_passengers_arr = num_passengers // 2
    num_passengers_dep = num_passengers // 2
    time_limit = 60  # Simulation window from 0 to 59 minutes

    failure_array = np.zeros(N_simulations)

    for sim in prange(N_simulations):
        num_vehicles = num_vehicles_initial
        platforms_available = np.zeros(num_platforms)  # Time each platform becomes available

        # Generate and sort operations
        arrival_times = generate_random_demand(num_passengers_arr, time_limit)
        departure_times = generate_random_demand(num_passengers_dep, time_limit)
        operations = np.append(arrival_times, departure_times)
        operation_type = np.append(np.ones(num_passengers_arr), np.zeros(num_passengers_dep))
        sort_idx = np.argsort(operations)
        operations = operations[sort_idx]
        operation_type = operation_type[sort_idx]

        for i in range(len(operations)):
            current_time = operations[i]
            current_type = operation_type[i]

            # Check platform availability
            earliest_platform = np.min(platforms_available)
            actual_time = max(current_time, earliest_platform)
            platform_wait = actual_time - current_time

            if platform_wait > max_wait_platform:
                failure_array[sim] = 1
                break

            # Update platform availability
            platform_idx = np.argmin(platforms_available)
            platforms_available[platform_idx] = actual_time + taxi_time

            # Handle resource constraints with waiting
            if current_type == 1:  # Arrival
                if num_vehicles + 1 > num_parkings:
                    # Find next departure that can free a parking spot
                    found = False
                    for j in range(i + 1, len(operations)):
                        if operation_type[j] == 0:
                            dep_time = operations[j]
                            wait_time = dep_time - actual_time
                            if wait_time <= max_wait_parking:
                                found = True
                                break
                    if not found:
                        failure_array[sim] = 1
                        break
                num_vehicles += 1
            else:  # Departure
                if num_vehicles - 1 < 0:
                    # Find next arrival that can provide a vehicle
                    found = False
                    for j in range(i + 1, len(operations)):
                        if operation_type[j] == 1:
                            arr_time = operations[j]
                            wait_time = arr_time - actual_time
                            if wait_time <= max_wait_departure:
                                found = True
                                break
                    if not found:
                        failure_array[sim] = 1
                        break
                num_vehicles -= 1

    return np.float32(np.mean(failure_array))



# -------------------------------
# Raster Data Loading Functions
# -------------------------------


def population_within_circle(x_coords, y_coords, pop_density, center_x, center_y, ds, radius=2000):
    """
    Calculate the total population within a circular area.

    Args:
        x_coords (1D array): X coordinates of the grid.
        y_coords (1D array): Y coordinates of the grid.
        pop_density (2D array): Population density values corresponding to the grid.
        center_x (float): X coordinate of the circle center.
        center_y (float): Y coordinate of the circle center.
        ds (float): Area per grid point (in km²).
        radius (float): Radius of the circle in meters.

    Returns:
        int: Sum of inhabitants within the circle.
    """
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    # Compute squared distances from the center
    distances = (X - center_x) ** 2 + (Y - center_y) ** 2
    # Boolean mask for points inside the circle
    inside_circle = distances <= radius**2
    inhabitants = np.sum(pop_density[inside_circle])
    return inhabitants





def read_data(filename):
    """
    Read raster data from a GeoTIFF file, compute coordinate grids, and convert to UTM.

    Args:
        filename (str): Path to the GeoTIFF file.

    Returns:
        tuple: (x array in UTM, y array in UTM, image data)
    """
    with rasterio.open(filename) as dataset:
        # Read the first band of the image
        image = dataset.read(1)
        
        # Get dataset bounds and affine transform
        bounds = dataset.bounds


        height, width = dataset.shape

        # Generate linspace vectors for longitude and latitude
        lon = np.linspace(bounds.left, bounds.right, width)
        lat = np.linspace(bounds.top, bounds.bottom, height)

    # Create a meshgrid of coordinates and convert to UTM
    lon2, lat2 = np.meshgrid(lon, lat)
    utm_data = utm.from_latlon(lat2, lon2)
    x2 = utm_data[0]
    y2 = utm_data[1]
    # Create uniform UTM coordinate grids based on the min and max values
    x = np.linspace(np.amin(x2), np.amax(x2), image.shape[1])
    y = np.linspace(np.amin(y2), np.amax(y2), image.shape[0])
    return x, y, image


# -------------------------------
# Main Simulation and Plotting
# -------------------------------


################################################################
############## Simulation parameters
taxi_time = 2  # Minimum time gap between consecutive operations
################################################################
# Vertiport location
lon_vertiport = -46.56210318814587
lat_vertiport = -23.54751562049972
# Influence radius in meters
radius = 9000 
#Frction of population that travel during an hour
fraction_of_passengers = 0.00006
#NUMBER_OF_PASSENGERS = POPULATION_INSIDE * fraction_of_passengers 


# Simulation settings
N_simulations = np.int32(5000)
num_platforms_vec = np.arange(1, 20).astype(int)
num_parkings_vec = np.arange(1, 50).astype(int)


#Binary variable to show or hide plots
plots = 1




#Script execution
# Convert the center coordinates to UTM
utm_center = utm.from_latlon(lat_vertiport, lon_vertiport)
x_vertiport = utm_center[0]
y_vertiport = utm_center[1]


#Waiting parameters
max_wait_platform = 1
max_wait_parking = 1
max_wait_departure = 1

# Read the raster data from the GeoTIFF file (assumed to be named "map_WGS.tif")
filename = "C:\\Users\\euqir\\OneDrive - Universidade de Vigo\\PHD UVigo\\Proyectos\\UAM Open Model DELFT\Tool\\map_WGS.tif"
x, y, image = read_data(filename)

# Discretization parameters for the mesh (in meters)
dx = 100
# Area of each grid point in km²
ds = dx**2 / 1e6

# Calculate the number of inhabitants within the influence circle
inhabitants_inside = population_within_circle(x, y, image, x_vertiport, y_vertiport, ds, radius)

print("The number of passengers is:", int(inhabitants_inside*fraction_of_passengers))

# Dummy call to compile the compute_failure_rate function
compute_failure_rate(inhabitants_inside, num_platforms_vec[0], num_parkings_vec[0],
                       np.int32(1), num_platforms_vec[0], fraction_of_passengers, taxi_time,max_wait_platform, max_wait_parking, max_wait_departure)

# Main simulation loop: compute failure rate for each combination of platforms and parkings
failure_matrix = np.zeros((len(num_platforms_vec), len(num_parkings_vec)))
for i in tqdm(range(len(num_platforms_vec))):
    for j in range(len(num_parkings_vec)):
        num_platforms = num_platforms_vec[i]
        num_parkings = num_parkings_vec[j]
        num_vehicles_initial = np.int32(num_parkings / 2)
        failure_matrix[i, j] = compute_failure_rate(inhabitants_inside, num_platforms, num_parkings,N_simulations, num_vehicles_initial, fraction_of_passengers, taxi_time, max_wait_platform, max_wait_parking, max_wait_departure)




# Save the failure matrix to an Excel file with headers.
df = pd.DataFrame(failure_matrix, index=num_platforms_vec, columns=num_parkings_vec)
df.index.name = "Number of Platforms"
df.columns.name = "Number of Parkings"
df.to_excel("failure_matrix.xlsx")
print("Failure matrix saved to 'failure_matrix.xlsx'.")


                                                     
if plots ==1:
    # Create a circle representing the influence area using Shapely
    circle = Point(x_vertiport, y_vertiport).buffer(radius)
    
    # Convert the circle to a GeoDataFrame for plotting
    circle_gdf = gpd.GeoDataFrame(geometry=[circle])

    # -------------------------------
    # Plotting the Raster Map and Influence Circle
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(image, cmap="coolwarm", extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
    plt.colorbar(im, ax=ax, label="Value")
    ax.scatter(x_vertiport, y_vertiport, color='red', marker='o', label="Vertiport")
    circle_gdf.boundary.plot(ax=ax, color='black', linewidth=2, label="Influence Radius")
    plt.xlabel("UTM X Coordinate")
    plt.ylabel("UTM Y Coordinate")
    plt.legend()
    plt.title("Sao Paulo Area")
    plt.show()
    
    # -------------------------------
    # Plotting the Failure Rate Matrix
    # -------------------------------
    plt.figure(figsize=(4, 3))
    plot = plt.pcolormesh(failure_matrix, cmap='gray', shading='auto')
    
    # Colorbar with label
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Expected Failure Rate (\%)', fontsize=13)
    
    # Axes labels
    plt.xlabel(r'Number of Parking Spots', fontsize=13)
    plt.ylabel(r'Number of Platforms', fontsize=13)
    
    # Set integer ticks for axes
    plt.xticks(ticks=np.arange(0.5, failure_matrix.shape[1] + 0.5, 5), labels=np.arange(1, failure_matrix.shape[1] + 1, 5))
    plt.yticks(ticks=np.arange(0.5, failure_matrix.shape[0] + 0.5, 5), labels=np.arange(1, failure_matrix.shape[0] + 1, 5))
    
    plt.tight_layout()
    plt.show()
