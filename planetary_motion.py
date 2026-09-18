from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from numerical_methods import (
    backward_difference,
    central_difference,
    forward_difference,
    second_central_difference,
)


# --------------------------------------------------
# Constants and output folders
# --------------------------------------------------

AU_TO_KM = 149_597_870.7
SECONDS_PER_DAY = 86_400

VELOCITY_CONVERSION = AU_TO_KM / SECONDS_PER_DAY
ACCELERATION_CONVERSION = AU_TO_KM * 1000 / SECONDS_PER_DAY**2

NUMBER_OF_POINTS = 400
DATA_DIRECTORY = Path("data")
FIGURE_DIRECTORY = Path("figures")


# a = semimajor axis in AU
# e = orbital eccentricity
# period = orbital period in days
PLANETS = {
    "Mercury": {"a": 0.387, "e": 0.2056, "period": 87.97, "color": "gray"},
    "Venus": {"a": 0.723, "e": 0.0068, "period": 224.70, "color": "orange"},
    "Earth": {"a": 1.000, "e": 0.0167, "period": 365.25, "color": "blue"},
    "Mars": {"a": 1.524, "e": 0.0934, "period": 686.98, "color": "red"},
    "Jupiter": {"a": 5.203, "e": 0.0489, "period": 4332.59, "color": "darkorange"},
    "Saturn": {"a": 9.537, "e": 0.0565, "period": 10759.22, "color": "goldenrod"},
    "Uranus": {"a": 19.191, "e": 0.0472, "period": 30688.5, "color": "deepskyblue"},
    "Neptune": {"a": 30.069, "e": 0.0086, "period": 60182.0, "color": "navy"},
}


SUMMARY_HEADER = [
    "planet",
    "semimajor_axis_AU",
    "eccentricity",
    "orbital_period_days",
    "perihelion_distance_AU",
    "aphelion_distance_AU",
    "maximum_speed_km_s",
    "minimum_speed_km_s",
    "maximum_acceleration_m_s2",
    "minimum_acceleration_m_s2",
    "forward_velocity_error_km_s",
    "backward_velocity_error_km_s",
    "central_velocity_error_km_s",
    "central_acceleration_error_m_s2",
]


CONVERGENCE_HEADER = [
    "number_of_points",
    "step_size_days",
    "forward_velocity_error_km_s",
    "backward_velocity_error_km_s",
    "central_velocity_error_km_s",
    "central_acceleration_error_m_s2",
]


# --------------------------------------------------
# Orbital calculations
# --------------------------------------------------

def solve_kepler(mean_anomaly, eccentricity, iterations=10):
    """Solve Kepler's equation using Newton's method."""
    eccentric_anomaly = mean_anomaly.copy()

    for _ in range(iterations):
        function = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        derivative = 1 - eccentricity * np.cos(eccentric_anomaly)
        eccentric_anomaly -= function / derivative

    return eccentric_anomaly


def generate_position(a, eccentricity, period, number_of_points):
    """Generate equally timed points along one elliptical orbit."""
    time = np.linspace(0, period, number_of_points, endpoint=False)
    step_size = period / number_of_points
    mean_anomaly = 2 * np.pi * time / period
    eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)

    x = a * (np.cos(eccentric_anomaly) - eccentricity)
    y = (
        a
        * np.sqrt(1 - eccentricity**2)
        * np.sin(eccentric_anomaly)
    )

    return time, step_size, eccentric_anomaly, x, y


def calculate_speed(x, y, step_size, difference_method):
    """Calculate orbital speed with a selected difference method."""
    vx = difference_method(x, step_size) * VELOCITY_CONVERSION
    vy = difference_method(y, step_size) * VELOCITY_CONVERSION
    return np.hypot(vx, vy)


def calculate_exact_speed(a, eccentricity, period, eccentric_anomaly):
    """Calculate the analytical orbital speed."""
    mean_motion = 2 * np.pi / period
    dE_dt = mean_motion / (
        1 - eccentricity * np.cos(eccentric_anomaly)
    )

    vx = (
        -a
        * np.sin(eccentric_anomaly)
        * dE_dt
        * VELOCITY_CONVERSION
    )
    vy = (
        a
        * np.sqrt(1 - eccentricity**2)
        * np.cos(eccentric_anomaly)
        * dE_dt
        * VELOCITY_CONVERSION
    )

    return np.hypot(vx, vy)


def calculate_numerical_acceleration(x, y, step_size):
    """Calculate acceleration with a second central difference."""
    ax = (
        second_central_difference(x, step_size)
        * ACCELERATION_CONVERSION
    )
    ay = (
        second_central_difference(y, step_size)
        * ACCELERATION_CONVERSION
    )

    return np.hypot(ax, ay)


def calculate_exact_acceleration(x, y, a, period):
    """Calculate acceleration from Newtonian gravity."""
    radius = np.hypot(x, y)
    mean_motion = 2 * np.pi / period
    gravitational_parameter = mean_motion**2 * a**3

    ax = (
        -gravitational_parameter
        * x
        / radius**3
        * ACCELERATION_CONVERSION
    )
    ay = (
        -gravitational_parameter
        * y
        / radius**3
        * ACCELERATION_CONVERSION
    )

    return np.hypot(ax, ay)


def calculate_orbit(planet, number_of_points=NUMBER_OF_POINTS):
    """Calculate position, speed, and acceleration for one planet."""
    a = planet["a"]
    eccentricity = planet["e"]
    period = planet["period"]

    time, step_size, eccentric_anomaly, x, y = generate_position(
        a,
        eccentricity,
        period,
        number_of_points,
    )

    return {
        "time": time,
        "x": x,
        "y": y,
        "speed_forward": calculate_speed(
            x, y, step_size, forward_difference
        ),
        "speed_backward": calculate_speed(
            x, y, step_size, backward_difference
        ),
        "speed_central": calculate_speed(
            x, y, step_size, central_difference
        ),
        "speed_exact": calculate_exact_speed(
            a,
            eccentricity,
            period,
            eccentric_anomaly,
        ),
        "acceleration_numerical": calculate_numerical_acceleration(
            x, y, step_size
        ),
        "acceleration_exact": calculate_exact_acceleration(
            x, y, a, period
        ),
    }


def calculate_all_orbits():
    """Calculate orbital data for all eight planets."""
    return {
        name: calculate_orbit(planet)
        for name, planet in PLANETS.items()
    }


def mean_absolute_error(numerical, exact):
    """Return the mean absolute error between two arrays."""
    return np.mean(np.abs(numerical - exact))


# --------------------------------------------------
# Summary table
# --------------------------------------------------

def create_summary_rows(orbit_data):
    """Create one summary-table row for each planet."""
    rows = []

    for name, planet in PLANETS.items():
        a = planet["a"]
        eccentricity = planet["e"]
        period = planet["period"]
        data = orbit_data[name]

        rows.append([
            name,
            a,
            eccentricity,
            period,
            a * (1 - eccentricity),
            a * (1 + eccentricity),
            np.max(data["speed_exact"]),
            np.min(data["speed_exact"]),
            np.max(data["acceleration_exact"]),
            np.min(data["acceleration_exact"]),
            mean_absolute_error(
                data["speed_forward"], data["speed_exact"]
            ),
            mean_absolute_error(
                data["speed_backward"], data["speed_exact"]
            ),
            mean_absolute_error(
                data["speed_central"], data["speed_exact"]
            ),
            mean_absolute_error(
                data["acceleration_numerical"],
                data["acceleration_exact"],
            ),
        ])

    return rows


def save_csv(path, header, rows):
    """Save a header and collection of rows to a CSV file."""
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def print_summary(rows):
    """Print a shortened planetary summary."""
    print("\nPlanetary orbital summary")
    print(
        f"{'Planet':<10}"
        f"{'Max speed':>14}"
        f"{'Min speed':>14}"
        f"{'Max accel.':>16}"
    )
    print(
        f"{'':<10}"
        f"{'(km/s)':>14}"
        f"{'(km/s)':>14}"
        f"{'(m/s^2)':>16}"
    )

    for row in rows:
        print(
            f"{row[0]:<10}"
            f"{row[6]:14.4f}"
            f"{row[7]:14.4f}"
            f"{row[8]:16.6e}"
        )


# --------------------------------------------------
# Plotting functions
# --------------------------------------------------

def format_axis(ax, xlabel, ylabel, title, legend_size=None):
    """Apply common labels and formatting to a plot."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    if legend_size is None:
        ax.legend()
    else:
        ax.legend(fontsize=legend_size)


def draw_orbits(ax, planet_names, orbit_data, show_time_points, title):
    """Draw a collection of planetary orbits on one axis."""
    for name in planet_names:
        x = orbit_data[name]["x"]
        y = orbit_data[name]["y"]
        color = PLANETS[name]["color"]

        # Repeat the first point to close the orbital curve.
        ax.plot(
            np.append(x, x[0]),
            np.append(y, y[0]),
            color=color,
            label=name,
        )

        if show_time_points:
            ax.scatter(x[::10], y[::10], color=color, s=10)

    ax.scatter(
        0,
        0,
        color="gold",
        edgecolor="black",
        s=150,
        label="Sun",
        zorder=5,
    )

    format_axis(
        ax,
        "x position (AU)",
        "y position (AU)",
        title,
        legend_size=8,
    )
    ax.set_aspect("equal")


def create_orbit_figure(orbit_data, show_time_points, title, filename):
    """Create a full-system panel and a terrestrial close-up panel."""
    all_planets = list(PLANETS)
    terrestrial_planets = ["Mercury", "Venus", "Earth", "Mars"]

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))

    draw_orbits(
        axes[0],
        all_planets,
        orbit_data,
        show_time_points,
        "Complete Solar System",
    )
    draw_orbits(
        axes[1],
        terrestrial_planets,
        orbit_data,
        show_time_points,
        "Terrestrial Planet Close-Up",
    )

    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / filename, dpi=300)


def create_velocity_figure(orbit_data):
    """Compare numerical and exact orbital speeds."""
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    mercury = orbit_data["Mercury"]
    mercury_time = mercury["time"] / PLANETS["Mercury"]["period"]

    axes[0].plot(
        mercury_time,
        mercury["speed_exact"],
        color="black",
        linewidth=3,
        label="Exact",
    )
    axes[0].plot(
        mercury_time,
        mercury["speed_forward"],
        "--",
        label="Forward",
    )
    axes[0].plot(
        mercury_time,
        mercury["speed_backward"],
        ":",
        label="Backward",
    )
    axes[0].plot(
        mercury_time,
        mercury["speed_central"],
        "-.",
        label="Central",
    )
    format_axis(
        axes[0],
        "Fraction of orbital period",
        "Orbital speed (km/s)",
        "Mercury: Numerical Method Comparison",
    )

    for name, planet in PLANETS.items():
        data = orbit_data[name]
        time_fraction = data["time"] / planet["period"]
        axes[1].plot(
            time_fraction,
            data["speed_exact"],
            color=planet["color"],
            label=name,
        )

    format_axis(
        axes[1],
        "Fraction of orbital period",
        "Orbital speed (km/s)",
        "Orbital Speeds of the Eight Planets",
        legend_size=8,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / "orbital_velocity_comparison.png",
        dpi=300,
    )


def create_acceleration_figure(orbit_data):
    """Compare numerical and exact orbital accelerations."""
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    mercury = orbit_data["Mercury"]
    mercury_time = mercury["time"] / PLANETS["Mercury"]["period"]

    axes[0].plot(
        mercury_time,
        mercury["acceleration_exact"],
        color="black",
        linewidth=3,
        label="Exact",
    )
    axes[0].plot(
        mercury_time,
        mercury["acceleration_numerical"],
        color="red",
        linestyle="--",
        label="Three-point central difference",
    )
    format_axis(
        axes[0],
        "Fraction of orbital period",
        "Acceleration (m/s²)",
        "Mercury: Numerical and Exact Acceleration",
    )

    for name, planet in PLANETS.items():
        data = orbit_data[name]
        time_fraction = data["time"] / planet["period"]
        axes[1].plot(
            time_fraction,
            data["acceleration_exact"],
            color=planet["color"],
            label=name,
        )

    axes[1].set_yscale("log")
    format_axis(
        axes[1],
        "Fraction of orbital period",
        "Acceleration (m/s²)",
        "Orbital Accelerations of the Eight Planets",
        legend_size=8,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / "orbital_acceleration_comparison.png",
        dpi=300,
    )


# --------------------------------------------------
# Convergence analysis
# --------------------------------------------------

def calculate_convergence():
    """Calculate Mercury's errors for progressively smaller steps."""
    mercury = PLANETS["Mercury"]
    point_counts = [50, 100, 200, 400, 800, 1600]
    rows = []

    for number_of_points in point_counts:
        data = calculate_orbit(mercury, number_of_points)
        step_size = mercury["period"] / number_of_points

        rows.append([
            number_of_points,
            step_size,
            mean_absolute_error(
                data["speed_forward"], data["speed_exact"]
            ),
            mean_absolute_error(
                data["speed_backward"], data["speed_exact"]
            ),
            mean_absolute_error(
                data["speed_central"], data["speed_exact"]
            ),
            mean_absolute_error(
                data["acceleration_numerical"],
                data["acceleration_exact"],
            ),
        ])

    return rows


def print_convergence(rows):
    """Print Mercury's velocity convergence results."""
    print("\nMercury convergence results")
    print(
        "Points   Step (days)   Forward error   "
        "Backward error   Central error"
    )

    for row in rows:
        print(
            f"{row[0]:6d}   "
            f"{row[1]:11.6f}   "
            f"{row[2]:13.6e}   "
            f"{row[3]:14.6e}   "
            f"{row[4]:13.6e}"
        )


def create_convergence_figure(rows):
    """Plot velocity and acceleration convergence for Mercury."""
    step_sizes = [row[1] for row in rows]
    forward_errors = [row[2] for row in rows]
    backward_errors = [row[3] for row in rows]
    central_errors = [row[4] for row in rows]
    acceleration_errors = [row[5] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].loglog(
        step_sizes,
        forward_errors,
        "o-",
        label="Forward difference",
    )
    axes[0].loglog(
        step_sizes,
        backward_errors,
        "s-",
        label="Backward difference",
    )
    axes[0].loglog(
        step_sizes,
        central_errors,
        "^-",
        label="Central difference",
    )
    axes[0].invert_xaxis()
    format_axis(
        axes[0],
        "Time step (days)",
        "Mean absolute speed error (km/s)",
        "Mercury Velocity Convergence",
    )
    axes[0].grid(True, which="both")

    axes[1].loglog(
        step_sizes,
        acceleration_errors,
        "o-",
        color="purple",
    )
    axes[1].set_xlabel("Time step (days)")
    axes[1].set_ylabel("Mean absolute acceleration error (m/s²)")
    axes[1].set_title("Mercury Acceleration Convergence")
    axes[1].invert_xaxis()
    axes[1].grid(True, which="both")

    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "error_convergence.png", dpi=300)


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():
    """Run the complete planetary-motion analysis."""
    DATA_DIRECTORY.mkdir(exist_ok=True)
    FIGURE_DIRECTORY.mkdir(exist_ok=True)

    orbit_data = calculate_all_orbits()

    summary_rows = create_summary_rows(orbit_data)
    save_csv(
        DATA_DIRECTORY / "planetary_summary.csv",
        SUMMARY_HEADER,
        summary_rows,
    )
    print_summary(summary_rows)

    create_orbit_figure(
        orbit_data,
        show_time_points=False,
        title="Elliptical Orbits of the Eight Planets",
        filename="planetary_orbits.png",
    )
    create_orbit_figure(
        orbit_data,
        show_time_points=True,
        title="Time-Sampled Elliptical Orbits of the Eight Planets",
        filename="time_sampled_orbits.png",
    )
    create_velocity_figure(orbit_data)
    create_acceleration_figure(orbit_data)

    convergence_rows = calculate_convergence()
    save_csv(
        DATA_DIRECTORY / "error_convergence.csv",
        CONVERGENCE_HEADER,
        convergence_rows,
    )
    print_convergence(convergence_rows)
    create_convergence_figure(convergence_rows)

    # Display all five figures simultaneously.
    plt.show()


if __name__ == "__main__":
    main()
