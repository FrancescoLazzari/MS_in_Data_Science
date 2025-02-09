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

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- ----------------------------------------------------------------------------------------------------------
--                                         Restructuring parameters                             
-- ----------------------------------------------------------------------------------------------------------

# Firstly, to verify if the database was of good quality, we checked if each table adhered to the criteria of 
# the Third Normal Form (3NF), according to which each table:
# - Is in first normal form (1NF) if each column contains only atomic values, and each column has a unique name
# - Is in second normal form (2NF) if it's already in 1NF, and all non-key attributes are fully functional 
#      dependent on the primary key
# - Is in third normal form (3NF) if it's already in 2NF, and there are no transitive dependencies

# Given that these conditions were already satisfied by our database, we then checked if the conditions for 
# Boyce-Codd Normal Form (BCNF) were also satisfied, which states:
# - For each non-trivial functional dependency, the determinant is a superkey

# In this case, too, the conditions were satisfied

# However, although the original database was already of good quality, we decided to implement the following improvements:

# 1) We created the table PLANE_MODEL, which identifies each possible combination of the three columns in 
#       the PLANES table that define an airplane model (manufacturer, model, type) with an index called 
#       'model_id' (primary key)
#     - PROS:
# 		 (a)   Lightens the PLANES table by removing three columns that occupied a lot of space due to their 
#                 VARCHAR datatype, thereby slightly reducing query execution time involving that table
# 		 (b)   Makes the database more robust to the insertion of new planes by avoiding the possibility of 
#                 assigning non-existent models to a plane
#     - CONS:  In the rare cases where a query needs the information now contained in the PLANE_MODEL table, 
#                 it now requires an additional JOIN, slightly increasing execution time, even if negligibly 
#                 (see Query 5 execution for a practical example)

# 2) We created a specific primary key for the table WEATHER to avoid having to use two columns ('date_hour' 
#       and 'airports_origin')as the primary key
#    - PROS:   Reduces the execution time during the JOINs between the FLIGHTS and WEATHER tables.
#    - CONSECUENCES:
# 		 (a)   Changing the primary key requires the manually inserting of the new foreign key column 
#                 'weather_id' into the FLIGHTS table through a specific Python script executed in the 
#                 preprocessing file
# 		 (b)   Removing the 'date_hour' column from the FLIGHTS table because it is no longer a foreign 
#                 key effectively results in the loss of the departure date for each record in that table 
#                 (the 'sched_dep' and 'sched_arr' columns by default only contain the time). 
#              Therefore, it was necessary to manually introduce that information into the ('sched_dep', 
#                 'sched_arr') columns, effectively changing their datatype from TIME to DATETIME. 
#              This modification was also carried out through a specific Python script executed in the 
#                  preprocessing file
# 		 (c)   Since the 'origin' column in FLIGHTS is no longer a foreign key for the weather column, 
#                  it was necessary to create a second relationship with the AIRPORTS table in order to 
#                  maintain that information in the FLIGHTS table (this relationship will require specific 
#                  integrity constraints)

# 3) We introduced integrity constraints through CHECK statements
#    - PROS:   Ensure that certain fields in certain columns could not have values that did not conform to 
#                  the meaning of the variable

# 4) We created specific PROCEDURES to automatically delete all records that did not meet certain conditions 
#       (see the relevant section for details) and a specific EVENT that executes these PROCEDURES automatically 
#       once a year
#    - PROS:   The presence of automations to systematically remove data from the database allows for keeping 
#                 the database size under control, given its function as a historical register, preventing it 
#                 from growing excessively creating slowdowns to maintain unnecessary data

# 5) We created some secondary index on coluns that are not primary key and are used often during the execution 
#       of some query (see the relevant section for the list of secondary index)
#    - PROS:   Slightly reduce the execution time of those query that involve that columns

-- -----------------------------------------------------
-- Schema DM_HW2                                    
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `DM_HW2` DEFAULT CHARACTER SET utf8mb3 ;

USE `DM_HW2` ;

-- -----------------------------------------------------
-- Table `DM_HW2`.`airlines`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW2`.`airlines` ;

CREATE TABLE IF NOT EXISTS `DM_HW2`.`airlines` (
  `carrier` VARCHAR(2) NOT NULL,
  `name` VARCHAR(30) NOT NULL,
  PRIMARY KEY (`carrier`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW2`.`airports`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW2`.`airports` ;

# In this table we added a integrity costrains on the column 'dst' 
# in order to prevent the insert of new records with non valid values on that field

CREATE TABLE IF NOT EXISTS `DM_HW2`.`airports` (
  `faa` VARCHAR(4) NOT NULL,
  `name` VARCHAR(55) NOT NULL,
  `time_zone` VARCHAR(25) NOT NULL,
  `tz` TINYINT NOT NULL,
  `dst` CHAR(1) NOT NULL CHECK (`dst` IN ('A', 'N')),
  `latitude` DECIMAL(10,7) NOT NULL,
  `longitude` DECIMAL(10,7) NOT NULL,
  `alt` SMALLINT NOT NULL,
  PRIMARY KEY (`faa`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW2`.`plane_model`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW2`.`plane_model` ;

CREATE TABLE IF NOT EXISTS `DM_HW2`.`plane_model` (
  `model_id` INT NOT NULL,
  `manufacturer` VARCHAR(45) NULL DEFAULT NULL,
  `model` VARCHAR(30) NULL DEFAULT NULL,
  `type` VARCHAR(35) NULL DEFAULT NULL,
  PRIMARY KEY (`model_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW2`.`planes`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW2`.`planes` ;

# In this table we added a integrity costrains on the column 'disused' 
# in order to prevent the insert of new records with non valid values on that field

CREATE TABLE IF NOT EXISTS `DM_HW2`.`planes` (
  `tail_num` VARCHAR(6) NOT NULL,
  `model_id` INT NOT NULL,
  `year` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `seats` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `speed` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `n_engines` TINYINT UNSIGNED NULL DEFAULT NULL,
  `engine_type` VARCHAR(30) NULL DEFAULT NULL,
  `disused` TINYINT NOT NULL CHECK (`disused` BETWEEN 0 AND 1),
  PRIMARY KEY (`tail_num`),
  CONSTRAINT `fk_plane_plane_model1`
    FOREIGN KEY (`model_id`)
    REFERENCES `DM_HW2`.`plane_model` (`model_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW2`.`weather`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW2`.`weather` ;

# In this table we added a integrity costrains on the foreign key 'airports_origin' 
# in order to prevent the insert of new records with non valid values on that field

CREATE TABLE IF NOT EXISTS `DM_HW2`.`weather` (
  `weather_id` INT NOT NULL,
  `date_hour` DATETIME NOT NULL,
  `airports_origin` VARCHAR(4) NOT NULL CHECK (`airports_origin` IN ('JFK', 'EWR', 'LGA')),
  `temp` DECIMAL(5,2) NULL DEFAULT NULL,
  `humid` DECIMAL(5,2) UNSIGNED NULL DEFAULT NULL,
  `precipitation` DECIMAL(5,2) UNSIGNED NULL DEFAULT NULL,
  `wind_speed` DECIMAL(9,5) UNSIGNED NULL DEFAULT NULL,
  `wind_dir` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `visibility` DECIMAL(4,2) UNSIGNED NULL DEFAULT NULL,
  `dewp` DECIMAL(4,2) NULL DEFAULT NULL,
  `pressure` DECIMAL(5,1) UNSIGNED NULL DEFAULT NULL,
  `wind_gust` DECIMAL(7,5) UNSIGNED NULL DEFAULT NULL,
  PRIMARY KEY (`weather_id`),
  CONSTRAINT `fk_weather_airports1`
    FOREIGN KEY (`airports_origin`)
    REFERENCES `DM_HW2`.`airports` (`faa`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW2`.`flights`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW2`.`flights` ;

# In this table we added a integrity costrains on the foreign key 'origin' 
# in order to prevent the insert of new records with non valid values on that field

CREATE TABLE IF NOT EXISTS `DM_HW2`.`flights` (
  `flight_id` INT NOT NULL AUTO_INCREMENT,
  `carrier` VARCHAR(2) NOT NULL,
  `tail_num` VARCHAR(6) NOT NULL,
  `weather_id` INT NOT NULL,
  `origin` VARCHAR(4) NOT NULL CHECK (`origin` IN ('JFK', 'EWR', 'LGA')),
  `destination` VARCHAR(3) NOT NULL,
  `sched_dep` DATETIME NOT NULL,
  `dep_delay_min` SMALLINT NULL DEFAULT NULL,
  `sched_arrive` DATETIME NULL DEFAULT NULL,
  `arr_delay_min` SMALLINT NULL DEFAULT NULL,
  `distance` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `air_time_min` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `canceled` TINYINT NOT NULL CHECK (`canceled` BETWEEN 0 AND 1),
  PRIMARY KEY (`flight_id`),
  CONSTRAINT `fk_flights_airlines`
    FOREIGN KEY (`carrier`)
    REFERENCES `DM_HW2`.`airlines` (`carrier`),
  CONSTRAINT `fk_flights_airports2`
    FOREIGN KEY (`destination`)
    REFERENCES `DM_HW2`.`airports` (`faa`),
  CONSTRAINT `fk_flights_airports3`
    FOREIGN KEY (`origin`)
    REFERENCES `DM_HW2`.`airports` (`faa`),
  CONSTRAINT `fk_flights_planes1`
    FOREIGN KEY (`tail_num`)
    REFERENCES `DM_HW2`.`planes` (`tail_num`),
  CONSTRAINT `fk_flights_weather1`
    FOREIGN KEY (`weather_id`)
    REFERENCES `DM_HW2`.`weather` (`weather_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;

-- -----------------------------------------------------
-- Index
-- -----------------------------------------------------

# The index for each primary/foreign key are automaticaly created so there is no need to explicitly create them

# In this schemas we also added some secondary index on the colum that are often used into queries on order to reduce the execution time

# NOTE:  - since we have a wetaher rilevation at the same hour for each departure airport we have 3 duplicated rows for each date_hour value
#        - in order to speed up the operation that involve this field we created a secondary index 
#        - the creation of this index was not possible into the schemas of HW1 since in was a primary key (with the airports_origin cloumn)
#          but in this version of the schemas we used 'weather_id' as primary key so we can create a secondary index on the 'date_hour' column
CREATE INDEX idx_date_hour ON DM_HW2.weather (date_hour);

# This was the only secondary index that we thought could speed up the execution time

-- -----------------------------------------------------
-- Procedures
-- -----------------------------------------------------

# Temporarily set the delimiter to // so that I can use the usual ; within the procedure
DELIMITER //

# Automatic delete of all the flights that are older than 5 years

CREATE PROCEDURE Delete_Old_Flights()
BEGIN
    DECLARE threshold DATETIME;
    SET threshold = DATE_SUB(NOW(), INTERVAL 5 YEAR);

    DELETE FROM DM_HW2.flights
    WHERE `sched_dep` < threshold;
END //

# Automatic delete of all the weather rilevations that are older than 5 years

CREATE PROCEDURE Delete_Old_Weather()
BEGIN
    DECLARE threshold DATETIME;
    SET threshold = DATE_SUB(NOW(), INTERVAL 5 YEAR);

    DELETE FROM DM_HW2.weather
    WHERE `date_hour` < threshold;
END //

# Automatic delete of all the planes, plane_model, airports and airlines that are no more useful for the records in the Database

CREATE PROCEDURE Delete_Unused_Planes_Airports_Airlines()
BEGIN
	# drop all the planes that doesn't have a flight
    DELETE FROM DM_HW2.planes
    WHERE tail_num NOT IN (SELECT DISTINCT tail_num FROM DM_HW2.flights);
    
    # drop all the plane models that are not used by at least one plane
    DELETE FROM DM_HW2.plane_model
    WHERE model_id NOT IN (SELECT DISTINCT model_id FROM DM_HW2.planes);
	
    # drop all the airports that doesn't have a flight
    DELETE FROM DM_HW2.airports
    WHERE faa NOT IN (SELECT DISTINCT origin FROM DM_HW2.flights);
	
    # drop all the airlines that doesn't have a flight
    DELETE FROM DM_HW2.airlines
    WHERE carrier NOT IN (SELECT DISTINCT carrier FROM DM_HW2.flights);
END //

# Reset the delimiter to its default value
DELIMITER ;

-- -----------------------------------------------------
-- Events
-- -----------------------------------------------------

DELIMITER //

CREATE EVENT IF NOT EXISTS CleanDataAnnually
ON SCHEDULE EVERY 1 YEAR
DO
BEGIN
    CALL Delete_Old_Flights();
    CALL Delete_Old_Weather();
    CALL Delete_Unused_Planes_Airports_Airlines();
END//

DELIMITER ;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;