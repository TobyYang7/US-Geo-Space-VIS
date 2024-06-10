# Airline Delay Visualization

This project visualizes airline delays across various airports in the United States. The visualization includes a map that highlights delay data, with the size of the airport markers representing passenger flow.

## Variables and Functions

### Variables

- `ax`: The matplotlib axes object where the map and plots are drawn.
- `time_slot`: The specific time slot for which the delay data is visualized. It is an integer value.
- `delay_type`: The type of delay to be visualized. It can be 'dep' for departure delay or 'arr' for arrival delay.
- `dropout_rate`: The rate at which connections between airports are randomly dropped to reduce the density of the visualization. It is a float value between 0 and 1.
- `od_delay`: A dictionary where the keys are tuples of airport codes (source, destination) and the values are the total delay times between these airports.
- `od_values`: A list of delay values extracted from the `od_delay` dictionary.
- `norm`: A normalization object used to scale delay values for visualization.
- `cmap`: The colormap used for visualizing delays.
- `sm`: A scalar mappable object used for the color bar.
- `cbar`: The color bar object that provides a legend for the delay values.

### Functions

#### `get_flow(airport_code)`

Reads the number of passengers for a given airport code from a CSV file.

- **Parameters**: `airport_code` (str) - The IATA code of the airport.
- **Returns**: The number of passengers for the specified airport.

#### `plot_map(ax, time_slot, delay_type='dep', dropout_rate=0.5)`

Plots the map with state boundaries, country boundaries, airport locations, and flight delay data.

- **Parameters**:
  - `ax`: The matplotlib axes object.
  - `time_slot`: The specific time slot for the delay data.
  - `delay_type`: The type of delay ('dep' or 'arr').
  - `dropout_rate`: The rate at which connections between airports are randomly dropped.

- **Returns**: The Basemap object.

#### `plot_airports(ax, map, s, time_slot, delay_type, dep=None, arr=None)`

Plots the airport locations on the map, with marker sizes representing passenger flow.

- **Parameters**:
  - `ax`: The matplotlib axes object.
  - `map`: The Basemap object.
  - `s`: The base size of the markers.
  - `time_slot`: The specific time slot for the delay data.
  - `delay_type`: The type of delay ('dep' or 'arr').
  - `dep`: Optional parameter for departure delay data.
  - `arr`: Optional parameter for arrival delay data.

- **Returns**: None.

#### `plot_edges(ax, map, od_delay)`

Plots the connections (edges) between airports on the map, with colors and line widths representing delay times.

- **Parameters**:
  - `ax`: The matplotlib axes object.
  - `map`: The Basemap object.
  - `od_delay`: A dictionary of delay times between airport pairs.

- **Returns**: None.
