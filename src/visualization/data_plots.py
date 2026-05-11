import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class DataPlots:

    def __init__(self, df_particles, df_shower, exp_path):

        self.df_particles = df_particles
        self.df_shower = df_shower

        self.exp_path = Path(exp_path)
        self.plot_path = self.exp_path / "data_plots"

        self.plot_path.mkdir(
            parents=True,
            exist_ok=True
        )

    # -------------------------------------------------
    # Particle energy distribution
    # -------------------------------------------------
    def particle_energy_histogram(self):

        plt.figure(figsize=(7,5))

        plt.hist(
            self.df_particles["energy"],
            bins=100
        )

        plt.xlabel("Energy (GeV)")
        plt.ylabel("Number of particles")

        plt.title(
            "Particle Energy Distribution"
        )

        plt.yscale("log")

        plt.tight_layout()

        plt.savefig(
            self.plot_path / "particle_energy_distribution.png"
        )

        plt.close()

    # -------------------------------------------------
    # Particles per shower
    # -------------------------------------------------
    def particles_per_shower(self):

        particles_per_shower = (
            self.df_particles
            .groupby("shower")
            .size()
        )

        plt.figure(figsize=(7,5))

        plt.hist(
            particles_per_shower,
            bins=30
        )

        plt.xlabel("Number of particles")
        plt.ylabel("Number of showers")

        plt.title(
            "Particles per Shower"
        )

        plt.yscale("log")

        plt.tight_layout()

        plt.savefig(
            self.plot_path / "particles_per_shower.png"
        )

        plt.close()

    # -------------------------------------------------
    # Momentum distributions
    # -------------------------------------------------
    def momentum_distribution(self):

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15,4)
        )

        momentum = ["px", "py", "pz"]

        for i, mom in enumerate(momentum):

            ax[i].hist(
                self.df_particles[mom],
                bins=100
            )

            ax[i].set_xlabel(
                f"{mom} (GeV/c)"
            )

            ax[i].set_ylabel(
                "Number of particles"
            )

            ax[i].set_title(
                f"{mom} Distribution"
            )

            ax[i].set_yscale("log")

        plt.tight_layout()

        plt.savefig(
            self.plot_path / "momentum_distribution.png"
        )

        plt.close()

    # -------------------------------------------------
    # Shower energy distribution
    # -------------------------------------------------
    def shower_energy_distribution(self):

        plt.figure(figsize=(7,5))

        plt.hist(
            self.df_shower["total_energy"],
            bins=30
        )

        plt.xlabel("Energy (GeV)")
        plt.ylabel("Number of showers")

        plt.title(
            "Shower Energy Distribution"
        )

        plt.yscale("log")

        plt.tight_layout()

        plt.savefig(
            self.plot_path / "shower_energy_distribution.png"
        )

        plt.close()

    # -------------------------------------------------
    # Angular distributions
    # -------------------------------------------------
    def angular_distribution(self):

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(12,4)
        )

        # Zenith
        ax[0].hist(
            self.df_shower["zenith"],
            bins=30
        )

        ax[0].set_xlabel(
            "Zenith (rad)"
        )

        ax[0].set_ylabel(
            "Number of showers"
        )

        ax[0].set_title(
            "Zenith Distribution"
        )

        # Azimuth
        ax[1].hist(
            self.df_shower["azimuth"],
            bins=30
        )

        ax[1].set_xlabel(
            "Azimuth (rad)"
        )

        ax[1].set_ylabel(
            "Number of showers"
        )

        ax[1].set_title(
            "Azimuth Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            self.plot_path / "angular_distribution.png"
        )

        plt.close()

    # -------------------------------------------------
    # General summary plot
    # -------------------------------------------------
    def summary_plot(self):

        fig, ax = plt.subplots(
            2,
            3,
            figsize=(16,10)
        )

        # Energy
        ax[0,0].hist(
            self.df_particles["energy"],
            bins=100
        )

        ax[0,0].set_title(
            "Particle Energy"
        )

        ax[0,0].set_yscale("log")

        # Particles per shower
        particles_per_shower = (
            self.df_particles
            .groupby("shower")
            .size()
        )

        ax[0,1].hist(
            particles_per_shower,
            bins=30
        )

        ax[0,1].set_title(
            "Particles per Shower"
        )

        ax[0,1].set_yscale("log")

        # Shower energy
        ax[0,2].hist(
            self.df_shower["total_energy"],
            bins=30
        )

        ax[0,2].set_title(
            "Shower Energy"
        )

        ax[0,2].set_yscale("log")

        # px
        ax[1,0].hist(
            self.df_particles["px"],
            bins=100
        )

        ax[1,0].set_title("px")

        ax[1,0].set_yscale("log")

        # py
        ax[1,1].hist(
            self.df_particles["py"],
            bins=100
        )

        ax[1,1].set_title("py")

        ax[1,1].set_yscale("log")

        # pz
        ax[1,2].hist(
            self.df_particles["pz"],
            bins=100
        )

        ax[1,2].set_title("pz")

        ax[1,2].set_yscale("log")

        plt.suptitle(
            "Dataset Summary",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            self.plot_path / "dataset_summary.png"
        )

        plt.close()

    # -------------------------------------------------
    # Run all plots
    # -------------------------------------------------
    def plot_all(self):

        self.particle_energy_histogram()

        self.particles_per_shower()

        self.momentum_distribution()

        self.shower_energy_distribution()

        self.angular_distribution()

        self.summary_plot()