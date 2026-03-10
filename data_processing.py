import pandas as pd


def clean_gdp_data(gdp):
    """
    Reshape GDP data into long format.
    """
    gdp_df = gdp.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name",
                 "Indicator Code"],
        var_name="Year",
        value_name="GDP"
    )
    gdp_df["Year"] = pd.to_numeric(gdp_df["Year"], errors="coerce")
    gdp_df = gdp_df.dropna(subset=["GDP", "Year"])
    return gdp_df[["Country Code", "Year", "GDP"]]


def clean_population_data(pop):
    """
    Reshape population data into long format.
    """
    pop_df = pop.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name",
                 "Indicator Code"],
        var_name="Year",
        value_name="Population"
    )
    pop_df["Year"] = pd.to_numeric(pop_df["Year"], errors="coerce")
    pop_df = pop_df.dropna(subset=["Population", "Year"])
    return pop_df[["Country Code", "Year", "Population"]]


def get_medals_per_year(olympics):
    """
    Return medal counts by year and country.
    """
    olympics_medals = olympics.dropna(subset=["Medal"])
    medals_per_year = (
        olympics_medals.groupby(["Year", "NOC"])
        .size()
        .reset_index(name="Medal_Count")
    )
    return medals_per_year


def merge_medals_and_gdp(medals_per_year, gdp_df):
    """
    Merge medal counts with GDP data.
    """
    return pd.merge(
        medals_per_year,
        gdp_df,
        left_on=["Year", "NOC"],
        right_on=["Year", "Country Code"],
        how="inner"
    )


def get_medal_totals(medals_per_year):
    """
    Return total medal counts by country.
    """
    medal_totals = (
        medals_per_year.groupby("NOC")["Medal_Count"]
        .sum()
        .sort_values(ascending=False)
    )
    return medal_totals


def get_top_countries_data(medals_per_year, medal_totals, top_n=10):
    """
    Return top countries and their medal data.
    """
    top_countries = medal_totals.head(top_n).index.tolist()
    medals_top = medals_per_year[medals_per_year["NOC"].isin(top_countries)]
    return top_countries, medals_top


def build_model_data(medals_per_year, gdp_df, pop_df):
    """
    Build a country-year dataset for regression.
    """
    model_df = pd.merge(
        medals_per_year,
        gdp_df,
        left_on=["Year", "NOC"],
        right_on=["Year", "Country Code"],
        how="left"
    )

    model_df = pd.merge(
        model_df,
        pop_df,
        left_on=["Year", "NOC"],
        right_on=["Year", "Country Code"],
        how="left",
        suffixes=("_gdp", "_pop")
    )

    model_df = model_df[["Year", "NOC", "Medal_Count", "GDP", "Population"]]
    model_df = model_df.sort_values(["NOC", "Year"])

    model_df["Previous_Medal_Count"] = (
        model_df.groupby("NOC")["Medal_Count"].shift(1)
    )

    model_df = model_df.dropna(
        subset=["GDP", "Population", "Previous_Medal_Count"]
    )

    return model_df