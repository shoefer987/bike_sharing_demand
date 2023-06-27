# Bike-Sharing Demand Prediction for Munich
This project uses data on the usage of public bike-sharing bikes together with weather data to predict the demand for
bike-sharing bikes in each district of Munich.

## Project Description
Two different data sources have been used to set up this project:

* [MVG](https://www.mvg.de/services/mvg-rad.html): usage of public MVG bike-sharing bikes from 2019 to 2022
* [Open-Meteo](https://open-meteo.com/): historical weather data for Munich on an hourly basis from 2019 to 2022

The aim was, to obtain an hourly prediction of the number of bike rentals for each district in Munich, taking into
account only the start time and location of rentals.

For this aim, an XGBoost Regressor is trained on the source data for each district. Predictions can be made through the
[API](https://github.com/shoefer987/bike_sharing_demand_api) by specifying a date within two weeks in the future. This
will trigger a query of Open-Meteo's weather forecasting API for the selected date which will be used for predicting the
demand.

## Credits
This project was conducted as 'final project' to finisch off the [Le Wagon Data Science Bootcamp Munich](https://www.lewagon.com/munich/data-science-course).
Many thanks to [Alex](https://github.com/azetxxx), [Archanaa](https://github.com/archanaakiruba), [Jonathan](https://github.com/Jonathan122802) and
[Jui](https://github.com/jui-kate) for the great teamwork!
