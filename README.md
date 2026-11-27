# Cascaded Hydropower Forecasting Service — Technical Manual & Service Specification v1.0

**Author:** Public Power Corporation
**Version:** 1.0
**Last updated:** 28 Jan 2026

---

## 1. Business Context & Definitions

The Cascaded Hydropower Forecasting Service provides integrated short-term forecasting for the Aliakmonas River hydropower cascade in Northern Greece, specifically targeting natural inflow prediction at Ilarionas HPP and production forecasting at Asomata HPP. The service addresses the critical challenge of forecasting in incomplete cascades where intermediate plant operational data (Polyfyto reservoir) is unavailable.

This service is designed to support operational and market-facing decision-making by hydropower operators, grid operators, and energy traders. Accurate and reliable forecasts are essential for energy scheduling and market bidding, water resource management, grid stability, and flood risk mitigation. The service is intended for continuous operational use, with defined availability, accuracy, and auditability requirements.

### Key terms

- **Hydropower Plant (HPP):** A power generation facility that converts hydraulic energy from water flow into electrical energy. In this context, plants are organized in a cascade structure.
- **Cascade Structure:** A series of interconnected hydropower plants where water flows sequentially through multiple facilities, i.e., Ilarionas → Polyfyto → Sfikia → Asomata → Agia Varvara → Makrochori.
- **Natural Inflow:** The volume of water entering a reservoir from its catchment area due to precipitation, snowmelt, and groundwater contributions, measured in m³/s or m³/day.
- **Reservoir-Based HPP:** A plant with significant storage capacity that can decouple water inflow from power generation, enabling temporal flexibility in operations (e.g., Ilarionas, Polyfyto, Asomata).
- **Pumped-Storage HPP:** A facility that can operate bidirectionally, i.e., generating power by releasing water downstream and consuming power by pumping water upstream (e.g., Sfikia).
- **Incomplete Cascade:** A cascade where operational data for one or more intermediate plants is unavailable, requiring implicit modelling of hidden dynamics.
- **Cascade Features:** Input variables derived from upstream plants that capture water flow dependencies across the cascade (e.g., lagged Ilarionas outflow, Sfikia pumped-back water).
- **Forecast Horizon:** The future time window for which predictions are generated (e.g., day-ahead: 24 hours, week-ahead: 7 days).
- **Temporal Resolution:** The time step at which forecasts are provided (e.g., daily aggregated production in MWh).
- **Irrigation Season:** The period of peak agricultural water demand (typically June-August in the Aliakmonas basin) when hydropower operations prioritize water delivery over energy optimization.
- **Specific Consumption:** The volume of water required to produce one unit of electrical energy, measured in m³/kWh. Lower values indicate higher efficiency.
- **Backtesting:** Rolling evaluation over historical time ranges to emulate real‑time usage.

### PPC (Plant Owner & Operator) Context

This service is intended for Public Power Corporation (PPC), the owner and operator of the Aliakmonas hydropower cascade, to strengthen operational planning and portfolio oversight by providing reliable day-ahead and week-ahead production forecasts with quantified uncertainty. PPC's primary objectives are to reduce imbalance exposure by minimizing deviations between scheduled and actual generation in day-ahead and intraday markets, optimize water resource allocation by balancing energy production with downstream irrigation demands and municipal water supply, and improve operational resilience by anticipating high inflow events for flood management while maintaining ecological flow requirements.

In day-to-day operations, PPC relies on forecast outputs to set production expectations for the next 24-48 hours, manage operational buffers and decide when to adjust reservoir release schedules, and coordinate cascade operations between HPPs to maximize system-wide value. The uncertainty information helps PPC anticipate periods of elevated hydrological risk and take preventive actions, especially during high inflow events that may require flood management or when downstream water demands constrain operational flexibility.

The service provides two integrated forecasting capabilities. Track 1 predicts natural inflows to Ilarionas reservoir from meteorological data with a 1-7 day horizon at daily temporal resolution, providing the upstream boundary condition for cascade modeling. Track 2 predicts electrical power generation at Asomata HPP using cascade dynamics with a 1-7 day horizon (day-ahead as primary use case) at daily resolution, supporting market bidding and operational planning. The key innovation addresses the “Polyfyto Challenge”, i.e., forecasting Asomata production despite missing operational data from the intermediate Polyfyto reservoir.

PPC supplies the necessary historical and real-time operational data (reservoir levels, production, water flows), while meteorological forecasts are retrieved from open-source weather models. Data handling expectations include strict time alignment, consistent plant identifiers, and clear documentation of inputs and assumptions used to generate each forecast. Performance is tracked using forecast accuracy metrics (R², MAE, RMSE, SMAPE), uncertainty calibration through confidence interval validation, and service reliability indicators covering API availability, response time, and data quality monitoring.

---

## 2. Problem Statement

The objective is to deliver a reliable, production-grade forecasting service for the Aliakmonas River hydropower cascade that predicts daily natural inflows to Ilarionas reservoir and daily electrical production at Asomata HPP over horizons of 1-7 days ahead. The service must operate continuously and provide forecasts that are consistent, traceable, and suitable for operational use across varying hydrological conditions, irrigation demands, and market participation requirements.

The service shall expose an authenticated interface for requesting forecasts per plant and time range, returning time-aligned outputs at daily temporal resolution. Forecasts must be generated with a clear issue time, include sufficient metadata to support auditability and reproducibility, and provide uncertainty quantification through prediction intervals.

The forecasting approach combines historical operational measurements, meteorological forecasts, and cascade features addressing two interconnected challenges. First, Ilarionas natural inflow forecasting must account for the Aliakmonas catchment's complex topography, including orographic precipitation effects, seasonal snowmelt dynamics, and travel time delays. Second, Asomata production forecasting must overcome the challenge of predicting generation despite missing operational data from the intermediate HPP. The service will be assessed as an operational capability, with emphasis on forecast quality, stability over time, and service reliability (availability and response performance).

---

## 3. Data Description

### 3.1 Data Dictionaries of HPPs

#### 3.1.1 Ilarionals HPP (Upstream)

| Variable | Variable name | Type | Measurement unit | Description | Allowed values / Examples |
|---|---|---|---|---|---|
| Plant Name | `hpp_name` | String | - | Plant identifier | "Ilarionas" |
| Year | `year` | Integer | - | Year of measurement | 2023 |
| Month | `month` | Integer | - | Month of measurement (1-12) | 6 |
| Day | `day` | Integer | - | Day of month | 15 |
| Date | `date` | Date | YYYY-MM-DD | Measurement date | 2023-06-15 |
| Reservoir Water Level | `reservoir_water_level_m` | Float | m (a.s.l) | Current reservoir elevation | 403.5 |
| Average Natural Inflow | `avg_natural_inflow_m3s` | Float | m³/s | Mean daily natural inflow rate | 15.3 |
| Daily Natural Inflow | `daily_natural_inflow_m3` | Float | m³ | Total daily natural inflow volume | 1,321,920 |
| HPP Production Water | `hpp_production_water_1000m3` | Float | 1000 m³ | Water consumed for power generation | 850 |
| Small HPP Production Water | `small_hpp_production_water_m3` | Float | m³ | Water used by auxiliary units | 5,200 |
| Overflow Outflow | `overflow_outflow_m3` | Float | m³ | Water released via spillway | 0 |
| Overflow (Aggregated) | `overflow_1000m3` | Float | 1000 m³ | Aggregated overflow volume | 0 |
| Weekend Indicator | `is_weekend` | Boolean | - | 1 if Saturday/Sunday, else 0 | 0 |

#### 3.1.2 Asomata HPP (Target Plant)

| Variable | Variable name | Type | Measurement unit | Description | Allowed values / Examples |
|---|---|---|---|---|---|
| Plant Name | `hpp_name` | String | - | Plant identifier | "Asomata" |
| Date | `date` | Date | YYYY-MM-DD | Measurement date | 2023-06-15 |
| Production Unit I | `production_unit_i_mwh` | Float | MWh | Unit 1 daily production | 650.2 |
| Production Unit II | `production_unit_ii_mwh` | Float | MWh | Unit 2 daily production | 648.7 |
| Total HPP Production | `production_hpp_total_mwh` | Float | MWh | Total daily production | 1,298.9 |
| Net Production | `net_production_mwh` | Float | MWh | Production minus auxiliary consumption | 1,295.3 |
| Peak Power | `peak_mw` | Float | MW | Maximum instantaneous power | 108.5 |
| Hours Operation Unit I | `hours_operation_unit_i` | Float | hours | Unit 1 operating hours | 23.5 |
| Hours Operation Unit II | `hours_operation_unit_ii` | Float | hours | Unit 2 operating hours | 23.2 |
| Total Hours Operation | `total_hours_operation` | Float | hours | Combined operating hours | 46.7 |
| Capacity Factor | `capacity_factor` | Float | - | Actual/max production ratio | 0.56 |
| Downtime Unit I - Maintenance (Plant) | `downtime_unit_i_maintenance_plant` | Float | hours | Planned maintenance | 0 |
| Downtime Unit I - Maintenance (External) | `downtime_unit_i_maintenance_outside` | Float | hours | Grid-related maintenance | 0 |
| Downtime Unit I - Failure (Plant) | `downtime_unit_i_failure_plant` | Float | hours | Equipment failure | 0 |
| Downtime Unit I - Failure (External) | `downtime_unit_i_failure_outside` | Float | hours | Grid failure | 0 |
| [Similar variables for Unit II] | `downtime_unit_ii_*` | Float | hours | Unit II downtime categories | 0 |
| Starts Unit I | `starts_unit_i` | Integer | count | Number of start-ups | 1 |
| Starts Unit II | `starts_unit_ii` | Integer | count | Number of start-ups | 1 |
| Reservoir Water Level | `reservoir_water_level_m` | Float | m (a.s.l) | Asomata reservoir elevation | 84.2 |
| Reservoir Water Content | `reservoir_water_content_1000m3` | Float | 1000 m³ | Stored water volume | 8,500 |
| Regulating Lake Water Level | `regulating_lake_water_level_m` | Float | m (a.s.l) | Daily regulating lake level | 52.8 |
| Regulating Lake Water Content | `regulating_lake_water_content_1000m3` | Float | 1000 m³ | Lake stored volume | 1,200 |
| Water Saved in Lake | `water_saved_in_lake_1000m3` | Float | 1000 m³ | Net daily storage change | 50 |
| Catchment Water Inflow | `catchment_water_inflow_1000m3` | Float | 1000 m³ | Local catchment inflow | 120 |
| Asomata Inflow | `asomata_inflow_1000m3` | Float | 1000 m³ | Total inflow to Asomata | 1,450 |
| Lake Inflow | `lake_inflow_1000m3` | Float | 1000 m³ | Inflow to regulating lake | 45 |
| Lake Losses | `lake_losses_1000m3` | Float | 1000 m³ | Evaporation and seepage | 8 |
| Regulating Lake Water Flow | `regulating_lake_water_flow_m3s` | Float | m³/s | Flow rate from regulating lake | 15.2 |
| Regulating Lake Total Water | `regulating_lake_total_water_1000m3` | Float | 1000 m³ | Total water in lake system | 1,250 |
| Production Water Consumption | `production_water_consumption_1000m3` | Float | 1000 m³ | Water used for generation | 1,299 |
| Specific Consumption | `specific_consumption` | Float | m³/kWh | Water per unit energy | 10.0 |
| Turbine Efficiency | `turbine_efficiency_pct` | Float | % | Conversion efficiency | 89.5 |
| Sfikia Production Water | `sfikia_production_water_1000m3` | Float | 1000 m³ | Water released from Sfikia → Asomata | 950 |
| Sfikia Pumped Upstream | `sfikia_pumped_upstream_1000m3` | Float | 1000 m³ | Water pumped back from Asomata → Sfikia | 150 |
| Weekend Indicator | `is_weekend` | Boolean | - | 1 if Saturday/Sunday | 0 |

#### 3.1.3 Weather Data (Open-Meteo API)

**Temporal Resolution:** Daily aggregated values

| Variable | Variable name | Type | Measurement unit | Description | Allowed values / Examples |
|---|---|---|---|---|---|
| Year | `year` | Integer | - | Year of measurement | 2023 |
| Month | `month` | Integer | - | Month of measurement (1-12) | 6 |
| Day | `day` | Integer | - | Day of month | 15 |
| Date | `date` | Date | YYYY-MM-DD | Measurement date | 2023-06-15 |
| Total Precipitation | `total_precipitation` | Float | mm/day | Rain and snow water equivalent | 12.5 |
| Rainfall | `rainfall` | Float | mm/day | Liquid precipitation only | 8.3 |
| Snowfall | `snowfall` | Float | mm/day | Snow water equivalent | 4.2 |
| Temperature 2m | `temperature_2m` | Float | °C | Air temperature at 2m height | 15.7 |
| Temperature Min | `temperature_min` | Float | °C | Daily minimum temperature | 8.2 |
| Temperature Max | `temperature_max` | Float | °C | Daily maximum temperature | 22.4 |
| Snow Depth | `snow_depth` | Float | cm | Snow pack depth | 45.0 |
| Snow Water Equivalent | `snow_water_equivalent` | Float | mm | Water content in snowpack | 180.0 |
| Solar Radiation | `solar_radiation` | Float | W/m² | Shortwave radiation for snowmelt energy balance | 450.3 |
| Soil Moisture | `soil_moisture` | Float | m³/m³ | Volumetric soil water content (0-10 cm depth) | 0.28 |
| Degree-Days | `degree_days` | Float | °C·day | Temperature sum for snowmelt modelling | 5.2 |
| Antecedent Precipitation Index | `api` | Float | mm | Weighted sum of past precipitation | 25.8 |

---

## 4. Analytics, Scope & Update Frequency

- **Temporal scope:** The service provides two integrated tracks that produce forecasts covering 1-7 days ahead with daily temporal resolution. Track 1 forecasts the Ilarionas natural inflow (m³/day) while Track 2 the Asomata Production (MWh/day).
- **Update frequency:** Forecasts are generated on demand via API requests.
- **Output format:** For each track and plant, the service returns a structured forecast package including:
  - **Issue time & validity window:** Timestamp (UTC) when the forecast is generated, plus the forecast horizon (1-7 days ahead).
  - **Time-indexed plant forecast:** A daily time series of predicted values with uncertainty quantification (10th, 50th, 90th percentiles).
  - **Traceability metadata:** Minimal identifiers (e.g., forecast/run ID and model version) so the forecast can be tracked in logs and reproduced if needed.

---

## 5. Evaluation Protocols & Metrics

The purpose of the evaluation is to verify that the service operates reliably and delivers consistent, accurate, and operationally usable hydropower production and inflow forecasts for the Aliakmonas cascade. Evaluation focuses on compliance with agreed metrics, service availability, and delivery of specified forecast outputs for each request.

### 5.1 Data Usage & Forecasting Protocol

- The service shall generate forecasts daily on a scheduled cadence or on demand via API, for horizons up to 7 days ahead of the stated issue time (UTC), aligned to daily temporal resolution.
- Only information available at or before the issue time may be used (e.g., telemetry up to that time and weather inputs available at that time)
- Forecast outputs shall be time-aligned and include a unique identifier and minimal traceability fields (e.g., run/forecast ID, model version).
- The service shall behave consistently under repeated requests for the same issue time and configuration.

### 5.2 Data Gaps and Exceptions

- Time periods with missing, invalid, or flagged measurements in actual production shall be excluded from forecast-quality scoring.
- If a plant has insufficient recent data to produce a forecast, the service shall return a clear, structured error (or documented fallback behaviour), and such events shall be counted under service reliability KPIs.

### 5.3 Service Evaluation Metrics & KPIs

The service shall be evaluated using the following quantitative metrics:

- **R² Score:** Coefficient of determination measuring variance explained by the model (0-1 scale, higher better), aggregated across forecast horizons and days.
- **MAE:** Mean Absolute Error of forecasted values versus actual measurements (MWh/day for production, m³/day for inflow), aggregated across forecast horizons and days.
- **RMSE:** Root Mean Squared Error of forecasted values versus actual measurements, aggregated across forecast horizons and days.
- **SMAPE:** Symmetric Mean Absolute Percentage Error providing scale-independent comparison across different production ranges and seasonal conditions.
- **Uncertainty Calibration:** Verification that stated confidence intervals are reliable by assessing whether prediction intervals contain observed values at the declared coverage rates (e.g., 80% intervals contain 80% of observations).
- **Horizon-Specific Accuracy:** Metrics calculated separately for each forecast day (D+1, D+2, ..., D+7) to assess forecast degradation with increasing horizon.
- **Availability:** Percentage of forecast requests that return a valid response within the agreed service levels.

---

## 6. Deliverables & Submissions

The selected provider shall deliver three  reports, aligned with the lifecycle of the service, together with the required technical specifications and deployment artefacts.

### 6.1 Deliverable Reports

Three delivery reports should be submitted for this service.

1. **Pre-Service Deliverable – Service Design & Setup Report**
   Submitted prior to service start, this report shall describe the proposed analytical approach, forecasting methodology, baseline definition, data requirements, system architecture, security measures, and integration plan. It shall also define the operational schedule and procedures for service execution.

2. **Intermediate Deliverable – Interim Performance & Operations Report**
   Submitted at an agreed midpoint of the service period, this report shall summarise service operation to date, data coverage, preliminary analytical results, and performance against the defined metrics and KPIs. Any issues, adaptations, or refinements to the methodology shall be documented.

3. **Final Deliverable – Final Evaluation & Recommendations Report**
   Submitted at the end of the service period, this report shall present final performance results and insights. The report shall also include guidance for future service continuation or scaling.

### 6.2 Technical Specifications & Submissions

- **Service Interface Documentation:** Full documentation of APIs, data formats, authentication, and access controls.
- **Deployment Artefacts:** The provider shall specify the deployment approach. Where containerisation is used, a Dockerfile or equivalent container specification shall be delivered to support reproducible deployment. Alternative deployment models shall be clearly documented.
- **Configuration, Versioning & Handover:** Documentation of configuration parameters, model and data versioning, and operational handover procedures.
- **Security & Data Protection Documentation:** Description of data handling, access control, and compliance with applicable data protection requirements.
