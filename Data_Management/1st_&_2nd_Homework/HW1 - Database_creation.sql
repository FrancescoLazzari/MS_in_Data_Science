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

-- -----------------------------------------------------
-- Schema DM_HW1
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `DM_HW1` DEFAULT CHARACTER SET utf8mb3 ;

USE `DM_HW1` ;

-- -----------------------------------------------------
-- Table `DM_HW1`.`airlines`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW1`.`airlines` ;

CREATE TABLE IF NOT EXISTS `DM_HW1`.`airlines` (
  `carrier` VARCHAR(2) NOT NULL,
  `name` VARCHAR(30) NOT NULL,
  PRIMARY KEY (`carrier`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW1`.`airports`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW1`.`airports` ;

CREATE TABLE IF NOT EXISTS `DM_HW1`.`airports` (
  `faa` VARCHAR(4) NOT NULL,
  `name` VARCHAR(55) NOT NULL,
  `time_zone` VARCHAR(25) NOT NULL,
  `tz` TINYINT NOT NULL,
  `dst` CHAR(1) NOT NULL,
  `latitude` DECIMAL(10,7) NOT NULL,
  `longitude` DECIMAL(10,7) NOT NULL,
  `alt` SMALLINT NOT NULL,
  PRIMARY KEY (`faa`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW1`.`planes`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW1`.`planes` ;

CREATE TABLE IF NOT EXISTS `DM_HW1`.`planes` (
  `tail_num` VARCHAR(6) NOT NULL,
  `year` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `manufacturer` VARCHAR(45) NULL DEFAULT NULL,
  `model` VARCHAR(30) NULL DEFAULT NULL,
  `type` VARCHAR(35) NULL DEFAULT NULL,
  `n_engines` TINYINT UNSIGNED NULL DEFAULT NULL,
  `seats` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `speed` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `engine_type` VARCHAR(30) NULL DEFAULT NULL,
  `disused` TINYINT NOT NULL,
  PRIMARY KEY (`tail_num`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW1`.`weather`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW1`.`weather` ;

CREATE TABLE IF NOT EXISTS `DM_HW1`.`weather` (
  `date_hour` DATETIME NOT NULL,   
  `airports_origin` VARCHAR(4) NOT NULL,
  `temp` DECIMAL(5,2) NULL DEFAULT NULL,
  `humid` DECIMAL(5,2) UNSIGNED NULL DEFAULT NULL,
  `precipitation` DECIMAL(5,2) UNSIGNED NULL DEFAULT NULL,
  `wind_speed` DECIMAL(9,5) UNSIGNED NULL DEFAULT NULL,
  `wind_dir` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `visibility` DECIMAL(4,2) UNSIGNED NULL DEFAULT NULL,
  `dewp` DECIMAL(4,2) NULL DEFAULT NULL,
  `pressure` DECIMAL(5,1) UNSIGNED NULL DEFAULT NULL,
  `wind_gust` DECIMAL(7,5) UNSIGNED NULL DEFAULT NULL,
  PRIMARY KEY (`date_hour`, `airports_origin`),
  CONSTRAINT `fk_weather_airports1`
    FOREIGN KEY (`airports_origin`)
    REFERENCES `DM_HW1`.`airports` (`faa`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `DM_HW1`.`flights`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `DM_HW1`.`flights` ;

CREATE TABLE IF NOT EXISTS `DM_HW1`.`flights` (
  `flight_ID` INT NOT NULL AUTO_INCREMENT,
  `carrier` VARCHAR(2) NOT NULL,
  `tail_num` VARCHAR(6) NOT NULL,
  `date_hour` DATETIME NOT NULL,    
  `origin` VARCHAR(4) NOT NULL,
  `destination` VARCHAR(3) NOT NULL,
  `sched_dep` TIME NOT NULL,
  `dep_delay_min` SMALLINT NULL DEFAULT NULL,
  `sched_arrive` TIME NULL DEFAULT NULL,
  `arr_delay_min` SMALLINT NULL DEFAULT NULL,
  `distance` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `air_time_min` SMALLINT UNSIGNED NULL DEFAULT NULL,
  `canceled` TINYINT NOT NULL,
  PRIMARY KEY (`flight_ID`),
  CONSTRAINT `fk_flights_airlines`
    FOREIGN KEY (`carrier`)
    REFERENCES `DM_HW1`.`airlines` (`carrier`),
  CONSTRAINT `fk_flights_airports2`
    FOREIGN KEY (`destination`)
    REFERENCES `DM_HW1`.`airports` (`faa`),
  CONSTRAINT `fk_flights_planes1`
    FOREIGN KEY (`tail_num`)
    REFERENCES `DM_HW1`.`planes` (`tail_num`),
  CONSTRAINT `fk_flights_weather1`   
    FOREIGN KEY (`date_hour`, `origin`)
    REFERENCES `DM_HW1`.`weather` (`date_hour` , `airports_origin`))  
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Index
-- -----------------------------------------------------

# The index for each primary/foreign key are automaticaly created so there is no need to explicitly create them

# We are not using secondary keys in this schemas


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;