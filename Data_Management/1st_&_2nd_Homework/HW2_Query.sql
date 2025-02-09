
# La Sapienza - University of Rome

## Accademic year 2023/2024

### EXAM: 
###   - Data Management 

### PROFESSORS: 
###   - Domenico Lembo
###   - Riccardo Rosati

### STUDENTS:
###   - Francesco Lazzari 1917922
###   - Riccardo Violano 2148833

-- --------------------------------------------------------------------------------------------------------
-- 						                        HW2_Q1						 
-- --------------------------------------------------------------------------------------------------------
# The most common route (faa + name) for each airline (name + carrier) considering only the departed and 
# not cancelled flights + the number of flights to that destination from that airline

# for the optimization we sobstituted the nested subquery with a temporany wiew created with WITH

# NOTE:  each temporany view could also be a permanent view but we preferred the first since we don't reuse
#        the same subquery into other querys but this is joust our choice and for the pourpose of the Homeworks
#        we also created at least one permanent view 
WITH AirlinesRouteCount AS (
    SELECT al.name AS Airline, al.carrier, destination, ap.name AS Airport, COUNT(*) AS num_flights
    FROM DM_HW2.flights f
    JOIN DM_HW2.airlines al ON al.carrier = f.carrier
    JOIN DM_HW2.airports ap ON ap.faa = f.destination
    WHERE f.canceled = 0
    GROUP BY al.carrier, destination
)
SELECT Airline, carrier, destination, Airport, num_flights
FROM AirlinesRouteCount
WHERE (carrier, num_flights) IN (
	# here we find the carrier that has also the max number of flights in the AirlinesRouteCount view
    SELECT carrier, MAX(num_flights)
    FROM AirlinesRouteCount
    GROUP BY carrier
)
ORDER BY num_flights DESC;

# OPTIMIZATION: ~ 48 %

# COMMENT:
# As expected the most common route for the airline company that departure from the thre NYC airports are 
# all in other US airports 

-- --------------------------------------------------------------------------------------------------------
-- 						                         HW2_Q2						 
-- --------------------------------------------------------------------------------------------------------

# For each airline company (carrier + name) return the Airplane (tail_num) that has the highest cumulative
# flight distance with that company + return the number of non canceled flights ad the maximun number of 
# passengers that the airplane can have have trasported

# NOTE:    if an airline company has two aiplanes that have the same highest cumulative flight they will  
#          be returned both

# NOTE 2:  this query could be optimized with the same structure as the one used in HW2_Q1 but we can also
#          avoid the subquery in the WHERE condition using a second WITH statement that find the MAX in the 
#          first temporary table 'TotalDistance'  
#          --> this second approach doesn't affect the computation time so is just an alternative sintax

WITH TotalDistance AS (
    SELECT 
		al.name AS Airline, 
		al.carrier, 
        f.tail_num AS Airplane, 
        SUM(distance) AS total_distance, 
        COUNT(*) AS N_Flights, 
        SUM(p.seats) AS Max_Cum_Passenger
    FROM DM_HW2.flights f
    JOIN DM_HW2.airlines al ON al.carrier = f.carrier
    JOIN DM_HW2.planes p ON p.tail_num = f.tail_num
	WHERE canceled = 0 
    GROUP BY carrier, p.tail_num
),    
MaxDistancePerCarrier AS (
    SELECT carrier, MAX(total_distance) AS max_distance
    FROM TotalDistance
    GROUP BY carrier
)
SELECT td.Airline, td.carrier, td.Airplane, td.total_distance, td.N_Flights, td.Max_Cum_Passenger
FROM TotalDistance td
JOIN MaxDistancePerCarrier mdpc ON td.carrier = mdpc.carrier AND td.total_distance = mdpc.max_distance
ORDER BY total_distance DESC;

# OPTIMIZATION: ~ 30 %

# COMMENT:
# There is a high variability between the airline companies which can be explained probably by a big difference
# in the number of flights that the airlines have accomplished -> that is confirmed by the number of flights 
# that the airplane have completed for that company

# In addition we can see that for the 'Mesa Airlines Inc.' there are two different (tail_num) airplanes that 
# have the same cumulative distance, number of flights and Max_num_Passengers with that company

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q3						 
-- --------------------------------------------------------------------------------------------------------

# For each destination airport in FLIGHTS return the month with the highest number of flights and it's flight count

# for the optimization we sobstituted the nested subquery with a permanent wiew created with WITH

# NOTE:  here we created a permanent view just to show the other possibility instead the temporany one
CREATE VIEW DM_HW2.DestinationMonthCount AS 
    SELECT 
        f.destination AS FAA,
        EXTRACT(MONTH FROM f.sched_dep) AS Peak_month,
        COUNT(*) AS Flight_count,
        # the DENSE_RANK function assign to each row (destination, month) a score based on the number of flight that each (destination, month) has
        # if different (destination, month) have the same number of flights they will recive the same score
        # NOTE: the simple RANK function would not give the same score to month that have the same number of flights
        DENSE_RANK() OVER (PARTITION BY f.destination ORDER BY COUNT(*) DESC) AS rank_num
    FROM DM_HW2.flights f
    GROUP BY f.destination, peak_month;
SELECT 
    a.name AS Airport, 
    dmc.FAA, 
    Peak_month, 
    Flight_count
FROM DM_HW2.DestinationMonthCount dmc
JOIN DM_HW2.airports a ON a.faa = dmc.FAA
# here we impose this because we want only the greatests for each destination
WHERE rank_num = 1
ORDER BY dmc.FAA, Peak_month DESC;

# OPTIMIZATION: ~ 98 %

# COMMENT:
# We have 104 different destinations airports in FLIGHTs but the query returned 134 rows.
# This is cause by the fact that 20 airports reach the maximum number of flights per month
# in more than one month. This is clearly visible sorting the rows by Airport

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q4						 
-- --------------------------------------------------------------------------------------------------------

# Return the average weather condition (temp, humid, precipitation, wind_speed, visibility) and the number 
# of canceled flights for each departure airport grouped by season and the number of canceled flights

# NOTE:  this version of the query has only a small reduction in the execution time of the one in HW1 but 
#        has a way more compacted ad easy to read sintax

# NOTE:  - since we have a wetaher rilevation at the same hour for each departure airport we have 3 duplicated rows for each date_hour value
#        - in order to speed up the operation that involve this field we created a secondary index 
#        - the creation of this index was not possible into the schemas of HW1 since in was a primary key (with the airports_origin cloumn)
#          but in this version of the schemas we used 'weather_id' as primary key so we can create a secondary index on the 'date_hour' column

# /!\  FOR THE CODE OF THE SECONDARY INDEX SEE THE FILE 'HW2_Database_creation' UNDER THE 'Index' SECTION

SELECT
    f.origin AS airport_origin,
    CASE
        WHEN MONTH(w.date_hour) IN (12, 1, 2) THEN 'Winter'
        WHEN MONTH(w.date_hour) IN (3, 4, 5) THEN 'Spring'
        WHEN MONTH(w.date_hour) IN (6, 7, 8) THEN 'Summer'
        WHEN MONTH(w.date_hour) IN (9, 10, 11) THEN 'Autumn'
    END AS Season,
    ROUND(AVG(w.temp),2) AS avg_temperature,
    ROUND(AVG(w.humid),2) AS avg_humidity,
    ROUND(AVG(w.precipitation),2) AS avg_precipitation,
    ROUND(AVG(w.wind_speed),2) AS avg_wind_speed,
    ROUND(AVG(w.visibility),2) AS avg_visibility,
    SUM(f.canceled) AS num_cancelled_flights
FROM DM_HW2.flights f
JOIN DM_HW2.weather w ON f.weather_id = w.weather_id
GROUP BY
    f.origin,
    CASE
        WHEN MONTH(w.date_hour) IN (12, 1, 2) THEN 'Winter'
        WHEN MONTH(w.date_hour) IN (3, 4, 5) THEN 'Spring'
        WHEN MONTH(w.date_hour) IN (6, 7, 8) THEN 'Summer'
        WHEN MONTH(w.date_hour) IN (9, 10, 11) THEN 'Autumn'
    END
ORDER BY Season;

# OPTIMIZATION: ~ 10 %

# COMMENTS:
# From the result we ca see that the average temperature doesn't vary too much (note: the temp is in F and not in C)
# between autumn and spring while obviously winter and summer are rispectively colder and hotter.
# Regarding the average precipitation we can say that in general in the NYC airports doesn't rain and have all a high
# visibility which slightly decrease in winter.
# Regarding the number of canceled flights we can sy that the season with less cancellation is the Autumn and the worst
# is the winter probably caused by worse meteo conditions 
# 

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q5						 
-- --------------------------------------------------------------------------------------------------------

# Return the Airplanes with (tailnum + manufacturer, model, type) of all the airplanes that have flight with 
# more that one airline company, the number of company which that plane has flight and theire name

# NOTE:   - this version of the query decrease the execution time of approximatly ~ 10 % 
#         - the reason of this decrease is due to the extra JOIN on the plane_model table
#         - however this solution contributed to decrease the execution time in all the query that
#           involve the PLANE table since now, removing the (manufacturer, model, type) colums
#           the table is smaller and faster
#           -  in addition this solution created a more robust dataset when we insert new airplanes
#              into the PLANE table
#           -  since also these 3 columns (manufacturer, model, type) are almost never used into 
#              queries and the increased computational time into the Q5 is negligible we prefer this solution

# NOTE 2: - the select on the tre column that now are in the PLANE_MODEL table (manufacturer, model, type) was 
#           not necessary for the output that could be optained only throught 'tail_num' in PLANES so thecnically 
#           we could obtain the same output without reducing the execution time passing from HW1 to HW2 
#           -  we chose to include also these fields in order to show how the downgrade is negligible compared to 
#              advantages given by this solution

SELECT
    f.tail_num,
    pm.manufacturer,
    pm.model,
    pm.type,
    COUNT(DISTINCT f.carrier) AS num_airlines,
    GROUP_CONCAT(DISTINCT f.carrier ORDER BY f.carrier) AS airline_codes,
    GROUP_CONCAT(DISTINCT a.name ORDER BY f.carrier) AS airline_names
FROM DM_HW2.flights f
JOIN DM_HW2.planes p USING (tail_num)
JOIN DM_HW2.plane_model pm USING (model_id)
JOIN DM_HW2.airlines a USING (carrier)
GROUP BY f.tail_num, pm.manufacturer, pm.model, pm.type
HAVING COUNT(DISTINCT f.carrier) > 1
ORDER BY num_airlines DESC;

# OPTIMIZATION: ~ -15 %

# COMMENT:
# We can see that only 17 Airplanes have flight with more than one company and all of these planes have 
# flight with 2 different companies. Regardless the companies we have 2 differnt couple of companies that 
# have used the same airplanes (Endeavor Air Inc. with Express Jet Airlines Inc. and Delta Airlines with 
# AirTran Airways Corporation). That can probably be explaned by a trading between the two couple of companies.

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q6						 
-- --------------------------------------------------------------------------------------------------------

# Return the 25 most old airplanes still in service (disused = 0) with some statistics like:
# - the year of construction 
#   -  exlude all the planes with NULL in the year field
# - total number of flights 
# - total distance 
# - average arrive delay
# - number of different airports in which that airplane has landed
# - number of different time zone in which that airplane has landed

SELECT f.tail_num, 
	   p.year,
       COUNT(*) AS num_flights,
	   COUNT(DISTINCT ap.tz) AS num_timezones,
       COUNT(DISTINCT f.destination) AS num_unique_destinations,
	   SUM(f.distance) AS total_distance,
	   ROUND(AVG(f.arr_delay_min),2) AS avg_arr_delay
FROM DM_HW2.flights f
JOIN DM_HW2.airports ap ON ap.faa = f.destination 
JOIN DM_HW2.planes p USING (tail_num)
WHERE p.year IS NOT NULL AND p.disused = 0  
GROUP BY f.tail_num, p.year
ORDER BY p.year ASC, total_distance DESC
LIMIT 25;

# OPTIMIZATION: negligible

# COMMENT:
# Surprisingly we have that the 25 oldest airplanes still in service are in the range that goes from year
# 1956 to year 1979

-- --------------------------------------------------------------------------------------------------------
-- 						                      HW2_Q7						 
-- --------------------------------------------------------------------------------------------------------

# Return for each destination airport the average arrive delay in that airport and the airlines companies 
# that have an average arrive delay bigger than 3 in that airport

# for the optimization we sobstituted the nested subquery with a temporany wiew created with WITH

# NOTE:  each temporany view could also be a permanent view but we preferred the first since we don't reuse
#        the same subquery into other querys but this is joust our choice and for the pourpose of the Homeworks
#        we also created at least one permanent view 
WITH CarrierDelayPerDestination AS (
    SELECT 
        al.name AS Airline,
        f.destination, 
        ROUND(AVG(arr_delay_min), 2) AS avg_arr_delay,
        # the DENSE_RANK function assign to each row airline a score based on the average delay that each airline has
        # if different airline have the same average delay they will recive the same score
        DENSE_RANK() OVER (PARTITION BY f.destination ORDER BY AVG(arr_delay_min) DESC) AS row_num
    FROM DM_HW2.flights f
    JOIN DM_HW2.airlines al USING (carrier)
    GROUP BY al.name, f.destination
    HAVING AVG(arr_delay_min) > 3
)
SELECT 
    ap.name AS Airport, 
    ap.faa AS FAA, 
    ROUND(AVG(f.arr_delay_min), 2) AS avg_airport_delay,
    (SELECT GROUP_CONCAT(Airline) FROM CarrierDelayPerDestination cd WHERE cd.destination = ap.faa) AS Airlines_with_high_delay
FROM DM_HW2.flights f
JOIN DM_HW2.airports ap ON ap.faa = f.destination
GROUP BY ap.faa
ORDER BY ap.faa;

# OPTIMIZATION: ~ 89 %

# COMMENT:
# As expected the average arrive delay in almost each airport is > 0 since the departure of one flight
# depends from a lot of factor that can easily create a delay
# The null results are given by airlines that doesn't have a delay bigger than 3 minutes

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q8						  
-- --------------------------------------------------------------------------------------------------------

# Return for all the flights (origin) that goes to a certan destination in a certan range of time and the 
# airline that offer that flight with the date and the hour-minute of the flight
 
SELECT
    f.origin,
    al.name AS airline_name,
    f.carrier AS airline_code,
    f.sched_dep 
FROM DM_HW2.flights f
JOIN DM_HW2.airlines al ON f.carrier = al.carrier
WHERE
    f.sched_dep BETWEEN '2013-06-01 00:00:00' AND '2013-06-07 23:59:59' 
    AND f.destination = 'LAX' -- Los Angeles
ORDER BY f.sched_dep, f.sched_dep ASC;

# OPTIMIZATION: constant time 

# COMMENT:
# This query even if is wery fast we think that is one of the query that will be executed the most since
# it's what a passenger will search when he want to book a flight so, considering it's importance for the 
# database the fast execution time is a good feature 
    
-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q9						 
-- --------------------------------------------------------------------------------------------------------

# Return the for each destination aiport and airline the count of flights affected by rain during the departure

SELECT
    al.name AS airline_name,
    f.carrier AS airline_code,
    ap.name AS destination_airport_name,
    f.destination AS destination_airport_code,
    COUNT(*) AS num_flights_with_rain
FROM DM_HW2.flights f
JOIN DM_HW2.airlines al USING (carrier)
JOIN DM_HW2.airports ap ON f.destination = ap.faa
JOIN DM_HW2.weather w USING (weather_id)
WHERE w.precipitation > 0 
GROUP BY al.name, f.carrier, ap.name, f.destination
ORDER BY num_flights_with_rain DESC;

# OPTIMIZATION: constant time 

# COMMENT:
# The result of this query is just a simple list on this condition

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW2_Q10						 
-- --------------------------------------------------------------------------------------------------------

# Returns all the destination airports that have a difference in the number of departures between the first 
# half and the second half of each year 

# NOTE:  the database contains only records for the year 2013 so that column of the query in this case is 
#        useless
  
WITH DepartureCounts AS (
	# Here we create a temporary view to calculate the departure counts for each (destination, year, month)
	SELECT
		destination AS FAA,
		YEAR(date_hour) AS year,
		MONTH(date_hour) AS month,
		COUNT(*) AS num_departures
	FROM DM_HW1.flights
	GROUP BY FAA, year, month
)
# Here we calculate the departure counts for the first half and second half for each (destination, year)
SELECT
	name AS Airports,
	dc.FAA,
    year,
    SUM(CASE WHEN month BETWEEN 1 AND 6 THEN num_departures ELSE 0 END) AS departures_first_half,
    SUM(CASE WHEN month BETWEEN 7 AND 12 THEN num_departures ELSE 0 END) AS departures_second_half,
    SUM(CASE WHEN month BETWEEN 7 AND 12 THEN num_departures ELSE 0 END) - SUM(CASE WHEN month BETWEEN 1 AND 6 THEN num_departures ELSE 0 END) AS departure_difference
FROM DepartureCounts dc
JOIN DM_HW2.airports ap ON ap.faa = dc.FAA
GROUP BY dc.FAA, year
HAVING departures_second_half < departures_first_half
ORDER BY departure_difference ASC;

# OPTIMIZATION: ~ 70 %

# COMMENT:
# Given the number of rows returned by the query, we can assert that the majority of airlines fly more 
# during the second half of the year (~40% or 40/104 fly less in the first half of the year). 
# In particular, if we narrow down the scope to airlines that have a negative difference considered significant 
# (delta > 30), then in that case, we would obtain only 26 airlines that meet that condition (~25% --> 26/104). 
# Remarkable is the decrease in some airlines such as 'Fort Lauderdale Hollywood Intl' or 'Southwest Florida Intl', 
# which even exceeds 500 flights.
