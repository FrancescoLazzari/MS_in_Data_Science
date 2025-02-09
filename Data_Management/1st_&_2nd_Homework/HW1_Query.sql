
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
-- 						                        HW1_Q1						 
-- --------------------------------------------------------------------------------------------------------
# The most common route (faa + name) for each airline (name + carrier) considering only the departed and 
# not cancelled flights + the number of flights to that destination from that airline

SELECT Airline, carrier, destination, Airport, num_flights
#We use a subquery to divide a big complex problem in more smaller ones
FROM (
	# Subquery to calculate the number of flights for each airline and destination
    SELECT al.name AS Airline, al.carrier, destination, ap.name AS Airport, COUNT(*) AS num_flights
    FROM DM_HW1.flights f
    JOIN DM_HW1.airlines al ON al.carrier = f.carrier
    JOIN DM_HW1.airports ap ON ap.faa = f.destination
    # we don't want the canceled flights
    WHERE f.canceled = 0
    # we group by carrier and destination because we want data for each airline
    GROUP BY al.carrier, destination

) AS AirlinesRouteCount
WHERE (carrier, num_flights) IN (
		# Subquery to find the maximum number of flights for each carrier
        SELECT carrier, MAX(num_flights)
        FROM (
			# here we find the number of flights for each carrier 
            SELECT al.carrier AS carrier, destination, COUNT(*) AS num_flights
            FROM DM_HW1.flights f
            JOIN DM_HW1.airlines al ON al.carrier = f.carrier
            JOIN DM_HW1.airports ap ON ap.faa = f.destination
            # we don't want the canceled flights
            WHERE f.canceled = 0
            GROUP BY al.carrier, destination
        ) AS AirlinesRouteCount
        GROUP BY carrier
)
#Ordering in a decrescent way so we have the best one in top
ORDER BY num_flights DESC;

# COMMENT:
# As expected the most common route for the airline company that departure from the thre NYC airports are 
# all in other US airports 

-- --------------------------------------------------------------------------------------------------------
-- 						                         HW1_Q2						 
-- --------------------------------------------------------------------------------------------------------

# For each airline company (carrier + name) return the Airplane (tail_num) that has the highest cumulative
# flight distance with that company + return the number of non canceled flights ad the maximun number of 
# passengers that the airplane can have have trasported

# NOTE:    if an airline company has two aiplanes that have the same highest cumulative flight they will  
#          be returned both

SELECT stat.Airline, stat.carrier, stat.Airplane, stat.Total_distance, stat.N_Flights, stat.Max_Cum_Passenger
FROM (
	# here we axtract for each airplane his airline and some statistics
    # like the total distance, the number of flights and the max cumulative passengers transported 
	SELECT 
        al.name AS Airline, 
        al.carrier, 
        f.tail_num AS Airplane, 
        SUM(distance) AS total_distance, 
        COUNT(*) AS N_Flights, 
        SUM(p.seats) AS Max_Cum_Passenger
    FROM DM_HW1.flights f
    JOIN DM_HW1.airlines al USING (carrier)
    JOIN DM_HW1.planes p USING (tail_num)
    # here we exclude the canceled flights from the count
    WHERE f.canceled = 0 
    # the order of the group by doesn't change the output
    GROUP BY f.tail_num, al.carrier
) AS stat
JOIN (
	# here we extract the airplane with the max total distance for each airlines
	SELECT carrier, MAX(total_distance) AS max_distance
    FROM (
		# here we extract the total distnce of each airplane and its airline
        # NOTE:  we exclude the canceled flights from the count
		SELECT f.carrier, SUM(distance) AS total_distance
        FROM DM_HW1.flights f
        WHERE f.canceled = 0 
        GROUP BY f.carrier, f.tail_num
	) AS PlanesTotalDistance
    # this group by is needed to pass from the total distance of each airplane to the 
    # plane with the max total distance for each airlines 
    GROUP BY carrier
# here we filter the airplane 'stat' leaving only the rows with the MAX(total_distance) (mdpc) for each carrier (airlines)
) AS mdpc ON stat.carrier = mdpc.carrier AND stat.total_distance = mdpc.max_distance
ORDER BY total_distance DESC;


# COMMENT:
# There is a high variability between the airline companies which can be explained probably by a big difference
# in the number of flights that the airlines have accomplished -> that is confirmed by the number of flights 
# that the airplane have completed for that company

# In addition we can see that for the 'Mesa Airlines Inc.' there are two different (tail_num) airplanes that 
# have the same cumulative distance, number of flights and Max_num_Passengers with that company

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q3						 
-- --------------------------------------------------------------------------------------------------------

# For each destination airport in FLIGHTS return the month with the highest number of flights and it's flight count

SELECT 
    a.name AS Airport,
    f1.destination AS FAA,
    MONTH(f1.date_hour) AS peak_month, 
    COUNT(*) AS flight_count
FROM DM_HW1.flights f1
JOIN DM_HW1.airports a ON a.faa = f1.destination
GROUP BY f1.destination, peak_month   # we have to extract this info from a datetime 
# We impose a condition for our result, in particular we want only the maximum number of flights for each destination, and then we want to know in which month 
HAVING
    COUNT(*) = ( SELECT MAX(count_inner) FROM (
                    SELECT COUNT(*) AS count_inner
                    FROM DM_HW1.flights f2
                    WHERE f2.destination = f1.destination
                    GROUP BY f2.destination, MONTH(f2.date_hour)
                    ) AS subquery
)
ORDER BY f1.destination, peak_month DESC;


# COMMENT:
# We have 104 different destinations airports in FLIGHTs but the query returned 134 rows.
# This is cause by the fact that 20 airports reach the maximum number of flights per month
# in more than one month. This is clearly visible sorting the rows by Airport

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q4						 
-- --------------------------------------------------------------------------------------------------------

# Return the average weather condition (temp, humid, precipitation, wind_speed, visibility) and the number 
# of canceled flights for each departure airport grouped by season and the number of canceled flights

# here we used a different query to extract the weather condition for each season and after we unite all the results
SELECT
    f.origin AS airport_origin, 'Winter' AS season, AVG(w.temp) AS avg_temperature, AVG(w.humid) AS avg_humidity,
    AVG(w.precipitation) AS avg_precipitation, AVG(w.wind_speed) AS avg_wind_speed, AVG(w.visibility) AS avg_visibility,
    SUM(f.canceled) AS num_cancelled_flights
FROM DM_HW1.flights f
JOIN DM_HW1.weather w ON f.date_hour = w.date_hour AND f.origin = w.airports_origin
WHERE MONTH(f.date_hour) IN (12, 1, 2)
GROUP BY f.origin

UNION ALL

SELECT
    f.origin AS airport_origin, 'Spring' AS season, AVG(w.temp) AS avg_temperature, AVG(w.humid) AS avg_humidity, 
    AVG(w.precipitation) AS avg_precipitation, AVG(w.wind_speed) AS avg_wind_speed, AVG(w.visibility) AS avg_visibility,
    SUM(f.canceled) AS num_cancelled_flights
FROM DM_HW1.flights f
JOIN DM_HW1.weather w ON f.date_hour = w.date_hour AND f.origin = w.airports_origin
WHERE MONTH(f.date_hour) IN (3, 4, 5)
GROUP BY f.origin

UNION ALL

SELECT
    f.origin AS airport_origin, 'Summer' AS season, AVG(w.temp) AS avg_temperature, AVG(w.humid) AS avg_humidity,
    AVG(w.precipitation) AS avg_precipitation, AVG(w.wind_speed) AS avg_wind_speed,
    AVG(w.visibility) AS avg_visibility, SUM(f.canceled) AS num_cancelled_flights
FROM DM_HW1.flights f
JOIN DM_HW1.weather w ON f.date_hour = w.date_hour AND f.origin = w.airports_origin
WHERE MONTH(f.date_hour) IN (6, 7, 8)
GROUP BY f.origin

UNION ALL

SELECT
    f.origin AS airport_origin, 'Autumn' AS season, AVG(w.temp) AS avg_temperature, AVG(w.humid) AS avg_humidity, 
    AVG(w.precipitation) AS avg_precipitation, AVG(w.wind_speed) AS avg_wind_speed, AVG(w.visibility) AS avg_visibility,
    SUM(f.canceled) AS num_cancelled_flights
FROM DM_HW1.flights f
JOIN DM_HW1.weather w ON f.date_hour = w.date_hour AND f.origin = w.airports_origin
WHERE MONTH(f.date_hour) IN (9, 10, 11)
GROUP BY f.origin;

# COMMENTS:
# From the result we ca see that the average temperature doesn't vary too much (note: the temp is in F and not in C)
# between autumn and spring while obviously winter and summer are rispectively colder and hotter.
# Regarding the average precipitation we can say that in general in the NYC airports doesn't rain and have all a high
# visibility which slightly decrease in winter.
# Regarding the number of canceled flights we can sy that the season with less cancellation is the Autumn and the worst
# is the winter probably caused by worse meteo conditions 
# 

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q5						 
-- --------------------------------------------------------------------------------------------------------

# Return the Airplanes with (tailnum + manufacturer, model, type) of all the airplanes that have flight with 
# more that one airline company, the number of company which that plane has flight and theire name

SELECT
    f.tail_num,
    p.manufacturer,
    p.model,
    p.type,
    COUNT(DISTINCT f.carrier) AS num_airlines,
    GROUP_CONCAT(DISTINCT f.carrier ORDER BY f.carrier) AS airline_codes,
    GROUP_CONCAT(DISTINCT a.name ORDER BY f.carrier) AS airline_names
FROM DM_HW1.flights f
JOIN DM_HW1.planes p USING (tail_num)
JOIN DM_HW1.airlines a USING (carrier)
GROUP BY f.tail_num, p.manufacturer, p.model, p.type
HAVING num_airlines > 1
ORDER BY num_airlines DESC;

# COMMENT:
# We can see that only 17 Airplanes have flight with more than one company and all of these planes have 
# flight with 2 different companies. Regardless the companies we have 2 differnt couple of companies that 
# have used the same airplanes (Endeavor Air Inc. with Express Jet Airlines Inc. and Delta Airlines with 
# AirTran Airways Corporation). That can probably be explaned by a trading between the two couple of companies.

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q6						 
-- --------------------------------------------------------------------------------------------------------

# Return the 25 most old airplanes still in service (disused = 0) with some statistics like:
# - the year of construction 
#   -  exclude all the planes with NULL in the year field
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
FROM DM_HW1.flights f
JOIN DM_HW1.airports ap ON ap.faa = f.destination 
JOIN DM_HW1.planes p ON f.tail_num = p.tail_num
WHERE p.year IS NOT NULL AND p.disused = 0  
GROUP BY f.tail_num, p.year
ORDER BY p.year ASC, total_distance DESC
LIMIT 25;

# COMMENT:
# Surprisingly we have that the 25 oldest airplanes still in service are in the range that goes from year
# 1956 to year 1979

-- --------------------------------------------------------------------------------------------------------
-- 						                      HW1_Q7						 
-- --------------------------------------------------------------------------------------------------------

# Return for each destination airport the average arrive delay in that airport and the airlines companies 
# that have an average arrive delay bigger than 3 in that airport

SELECT 
    ap.name AS Airport, 
    ap.faa AS FAA, 
    ROUND(AVG(f.arr_delay_min), 2) AS avg_airport_delay,
    # We use a subquery to divide the problem
    (    SELECT GROUP_CONCAT(al.name  ORDER BY AVG(f.arr_delay_min) DESC)
        FROM DM_HW1.airlines al
        WHERE al.carrier IN (
        # we  are calculating the average delay, and we are keeping only the information of airlines that have a delay bigger than 3 minutes
        # this because we want to know just which airlines we should avoid
            SELECT f2.carrier
            FROM DM_HW1.flights f2
            WHERE f2.destination = ap.faa
            GROUP BY f2.carrier
            HAVING AVG(f2.arr_delay_min) > 3
            ORDER BY AVG(f2.arr_delay_min) DESC
        )
    ) AS Airlines_with_high_delay
FROM DM_HW1.flights f
JOIN DM_HW1.airports ap ON ap.faa = f.destination
GROUP BY ap.faa
ORDER BY ap.faa ASC;
    
# COMMENT:
# As expected the average arrive delay in almost each airport is > 0 since the departure of one flight
# depends from a lot of factor that can easily create a delay
# The null results are given by airlines that doesn't have a delay bigger than 3 minutes

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q8						  
-- --------------------------------------------------------------------------------------------------------

# Return for all the flights (origin) that goes to a certan destination in a certan range of time and the 
# airline that offer that flight with the date and the hour-minute of the flight

SELECT
    f.origin,
    al.name AS airline_name,
    f.carrier AS airline_code,
    DATE(f.date_hour) AS date,
    f.sched_dep
FROM DM_HW1.flights f
JOIN DM_HW1.airlines al ON f.carrier = al.carrier
WHERE
    f.date_hour BETWEEN '2013-06-01 00:00:00' AND '2013-06-07 23:59:59' 
    AND f.destination = 'LAX' -- Los Angeles
ORDER BY DATE(f.date_hour), f.sched_dep ASC;

# COMMENT:
# This query even if is wery fast we think that is one of the query that will be executed the most since
# it's what a passenger will search when he want to book a flight so, considering it's importance for the 
# database the fast execution time is a good feature 


-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q9						 
-- --------------------------------------------------------------------------------------------------------

# Return the for each destination aiport and airline the count of flights affected by rain during the departure

SELECT
    al.name AS airline_name,
    f.carrier AS airline_code,
    ap.name AS destination_airport_name,
    f.destination AS destination_airport_code,
    COUNT(*) AS num_flights_with_rain
FROM DM_HW1.flights f
JOIN DM_HW1.airlines al ON f.carrier = al.carrier
JOIN DM_HW1.airports ap ON f.destination = ap.faa
JOIN DM_HW1.weather w ON f.date_hour = w.date_hour AND f.origin = w.airports_origin
WHERE w.precipitation > 0 
GROUP BY al.name, f.carrier, ap.name, f.destination
ORDER BY num_flights_with_rain DESC;

# COMMENT:
# The result of this query is just a simple list on this condition

-- --------------------------------------------------------------------------------------------------------
-- 						                          HW1_Q10						 
-- --------------------------------------------------------------------------------------------------------

# Returns all the destination airports that have a difference in the number of departures between the first 
# half and the second half of each year 

# NOTE:  the database contains only records for the year 2013 so that column of the query in this case is 
#        useless

SELECT
	name AS Airport,
    destination AS FAA,
    YEAR(date_hour) AS year,
    SUM(CASE WHEN MONTH(date_hour) BETWEEN 1 AND 6 THEN 1 ELSE 0 END) AS departures_first_half,
    SUM(CASE WHEN MONTH(date_hour) BETWEEN 7 AND 12 THEN 1 ELSE 0 END) AS departures_second_half,
    SUM(CASE WHEN MONTH(date_hour) BETWEEN 7 AND 12 THEN 1 ELSE 0 END) - SUM(CASE WHEN MONTH(date_hour) BETWEEN 1 AND 6 THEN 1 ELSE 0 END) AS departure_difference
FROM DM_HW1.flights f
JOIN DM_HW1.airports ap ON ap.faa = f.destination
GROUP BY FAA, year
HAVING departures_second_half < departures_first_half
ORDER BY departure_difference ASC;


# COMMENT:
# Given the number of rows returned by the query, we can assert that the majority of airlines fly more 
# during the second half of the year (~40% or 40/104 fly less in the first half of the year). 
# In particular, if we narrow down the scope to airlines that have a negative difference considered significant 
# (delta > 30), then in that case, we would obtain only 26 airlines that meet that condition (~25% --> 26/104). 
# Remarkable is the decrease in some airlines such as 'Fort Lauderdale Hollywood Intl' or 'Southwest Florida Intl', 
# which even exceeds 500 flights.
