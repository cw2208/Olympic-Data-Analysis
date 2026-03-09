import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import clean_gdp_data
from data_processing import clean_population_data
from data_processing import get_medals_per_year
from data_processing import merge_medals_and_gdp
from data_processing import get_medal_totals
from data_processing import get_top_countries_data


def plot_gdp_vs_medals(merged_data):
    """
    Plot GDP versus Olympic medal count.
    """
    plt.figure(figsize=(12, 8))
    sns.regplot(
        data=merged_data,
        x="GDP",
        y="Medal_Count",
        line_kws={"color": "red"}
    )
    plt.xscale("log")
    plt.title("GDP vs Olympic Medal Count")
    plt.xlabel("GDP")
    plt.ylabel("Medal Count")
    plt.grid(True)
    plt.savefig("results/gdp_vs_medals.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_medals_over_time(medal_data, countries):
    """
    Plot medal counts over time for top countries.
    """
    plt.figure(figsize=(15, 8))
    sns.lineplot(
        data=medal_data,
        x="Year",
        y="Medal_Count",
        hue="NOC",
        hue_order=countries
    )
    plt.title("Olympic Medals Over Time (Top Countries)")
    plt.xlabel("Year")
    plt.ylabel("Medal Count")
    plt.grid(True)
    plt.savefig(
        "results/top10_medals_over_time.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_gdp_over_time(gdp_df, countries):
    """
    Plot GDP trends for top medal countries.
    """
    data = gdp_df[gdp_df["Country Code"].isin(countries)]

    plt.figure(figsize=(15, 8))
    sns.lineplot(data=data, x="Year", y="GDP", hue="Country Code")
    plt.yscale("log")
    plt.title("GDP of Top Medal Countries Over Time")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.savefig("results/top10_gdp_over_time.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_population_over_time(pop_df, countries):
    """
    Plot population trends for top medal countries.
    """
    data = pop_df[pop_df["Country Code"].isin(countries)]

    plt.figure(figsize=(15, 8))
    sns.lineplot(data=data, x="Year", y="Population", hue="Country Code")
    plt.yscale("log")
    plt.title("Population of Top Medal Countries Over Time")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.savefig("results/top10_population_over_time.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_age_distribution(olympics):
    """
    Plot the distribution of athlete ages.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(olympics["Age"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Athlete Ages")
    plt.xlabel("Age")
    plt.ylabel("Athletes")
    plt.savefig("results/athlete_age_distribution.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_global_gdp(gdp_df):
    """
    Plot total global GDP over time.
    """
    global_gdp = gdp_df.groupby("Year")["GDP"].sum().reset_index()

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=global_gdp, x="Year", y="GDP")
    plt.yscale("log")
    plt.title("Global GDP Over Time")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.savefig("results/global_gdp_over_time.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_global_population(pop_df):
    """
    Plot total global population over time.
    """
    global_pop = pop_df.groupby("Year")["Population"].sum().reset_index()

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=global_pop, x="Year", y="Population")
    plt.yscale("log")
    plt.title("Global Population Over Time")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.savefig("results/global_population_over_time.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_medal_distribution(medal_totals):
    """
    Plot medal share by country.
    """
    top_n = 20
    medal_counts = medal_totals.head(top_n).copy()
    other = medal_totals.iloc[top_n:].sum()

    if other > 0:
        medal_counts.loc["Other"] = other

    plt.figure(figsize=(10, 10))
    plt.pie(
        medal_counts.values,
        labels=medal_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Share of Olympic Medals by Country")
    plt.savefig("results/medal_share_pie_chart.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def main():
    olympics = pd.read_csv("data/athlete_events.csv")
    gdp = pd.read_csv("data/gdp.csv")
    pop = pd.read_csv("data/pop.csv")

    gdp_df = clean_gdp_data(gdp)
    pop_df = clean_population_data(pop)

    medals_per_year = get_medals_per_year(olympics)
    merged_data = merge_medals_and_gdp(medals_per_year, gdp_df)
    medal_totals = get_medal_totals(medals_per_year)
    top_countries, medals_top = get_top_countries_data(
        medals_per_year, medal_totals
    )

    plot_gdp_vs_medals(merged_data)
    plot_medals_over_time(medals_top, top_countries)
    plot_gdp_over_time(gdp_df, top_countries)
    plot_population_over_time(pop_df, top_countries)
    plot_age_distribution(olympics)
    plot_global_gdp(gdp_df)
    plot_global_population(pop_df)
    plot_medal_distribution(medal_totals)


if __name__ == "__main__":
    main()