import pandas as pd

from data_processing import clean_gdp_data
from data_processing import clean_population_data
from data_processing import get_medals_per_year
from data_processing import merge_medals_and_gdp
from data_processing import get_medal_totals
from data_processing import get_top_countries_data


def test_clean_gdp_data():
    """Test GDP data cleaning."""
    gdp = pd.DataFrame({
        "Country Name": ["United States", "Canada"],
        "Country Code": ["USA", "CAN"],
        "Indicator Name": ["GDP", "GDP"],
        "Indicator Code": ["NY.GDP", "NY.GDP"],
        "2000": [100.0, 50.0],
        "2001": [110.0, None]
    })

    result = clean_gdp_data(gdp)

    assert list(result.columns) == ["Country Code", "Year", "GDP"]
    assert len(result) == 3
    assert "USA" in result["Country Code"].values
    assert 2000 in result["Year"].values


def test_clean_population_data():
    """Test population data cleaning."""
    pop = pd.DataFrame({
        "Country Name": ["United States"],
        "Country Code": ["USA"],
        "Indicator Name": ["Population"],
        "Indicator Code": ["SP.POP"],
        "2000": [300.0],
        "2001": [305.0]
    })

    result = clean_population_data(pop)

    assert list(result.columns) == ["Country Code", "Year", "Population"]
    assert len(result) == 2
    assert result.iloc[0]["Country Code"] == "USA"


def test_get_medals_per_year():
    """Test medal counts by year and country."""
    olympics = pd.DataFrame({
        "Year": [2000, 2000, 2004, 2004],
        "NOC": ["USA", "USA", "CHN", "USA"],
        "Medal": ["Gold", "Silver", None, "Bronze"]
    })

    result = get_medals_per_year(olympics)

    assert len(result) == 2
    assert "Medal_Count" in result.columns
    usa_2000 = result[(result["Year"] == 2000) & (result["NOC"] == "USA")]
    assert usa_2000.iloc[0]["Medal_Count"] == 2


def test_merge_medals_and_gdp():
    """Test merging medal and GDP data."""
    medals_per_year = pd.DataFrame({
        "Year": [2000, 2004],
        "NOC": ["USA", "CHN"],
        "Medal_Count": [10, 8]
    })

    gdp_df = pd.DataFrame({
        "Country Code": ["USA", "CHN"],
        "Year": [2000, 2004],
        "GDP": [1000.0, 900.0]
    })

    result = merge_medals_and_gdp(medals_per_year, gdp_df)

    assert len(result) == 2
    assert "GDP" in result.columns
    assert "Country Code" in result.columns


def test_get_medal_totals():
    """Test total medal counts by country."""
    medals_per_year = pd.DataFrame({
        "Year": [2000, 2004, 2000],
        "NOC": ["USA", "USA", "CHN"],
        "Medal_Count": [10, 8, 7]
    })

    result = get_medal_totals(medals_per_year)

    assert result["USA"] == 18
    assert result["CHN"] == 7


def test_get_top_countries_data():
    """Test filtering top countries medal data."""
    medals_per_year = pd.DataFrame({
        "Year": [2000, 2004, 2000],
        "NOC": ["USA", "USA", "CHN"],
        "Medal_Count": [10, 8, 7]
    })

    medal_totals = pd.Series({"USA": 18, "CHN": 7, "RUS": 5})

    top_countries, medals_top = get_top_countries_data(
        medals_per_year, medal_totals, top_n=2
    )

    assert top_countries == ["USA", "CHN"]
    assert set(medals_top["NOC"]) == {"USA", "CHN"}


def main():
    test_clean_gdp_data()
    test_clean_population_data()
    test_get_medals_per_year()
    test_merge_medals_and_gdp()
    test_get_medal_totals()
    test_get_top_countries_data()


if __name__ == "__main__":
    main()