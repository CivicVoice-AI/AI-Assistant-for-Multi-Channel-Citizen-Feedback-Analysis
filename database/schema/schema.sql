-- ============================================================
-- CIVICVOICE DATABASE SCHEMA
-- ============================================================

CREATE DATABASE IF NOT EXISTS civicvoice_db;

USE civicvoice_db;


-- ============================================================
-- 1. SOURCES
-- ============================================================

CREATE TABLE IF NOT EXISTS sources (
    source_id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(50) NOT NULL UNIQUE
);


-- ============================================================
-- 2. LOCATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS locations (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    location_name VARCHAR(150) NOT NULL UNIQUE
);


-- ============================================================
-- 3. FEEDBACK
-- ============================================================

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id VARCHAR(50) PRIMARY KEY,

    source_id INT NOT NULL,

    location_id INT,

    feedback_timestamp DATETIME,

    feedback_text TEXT NOT NULL,

    organization VARCHAR(255),

    text_clean TEXT,

    FOREIGN KEY (source_id)
        REFERENCES sources(source_id),

    FOREIGN KEY (location_id)
        REFERENCES locations(location_id),

    INDEX idx_source (source_id),
    INDEX idx_location (location_id),
    INDEX idx_timestamp (feedback_timestamp)
);


-- ============================================================
-- 4. FEEDBACK URGENCY
-- ============================================================

CREATE TABLE IF NOT EXISTS feedback_urgency (
    urgency_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    feedback_id VARCHAR(50) NOT NULL,

    urgency_label VARCHAR(20) NOT NULL,

    FOREIGN KEY (feedback_id)
        REFERENCES feedback(feedback_id)
        ON DELETE CASCADE,

    UNIQUE (feedback_id),

    INDEX idx_urgency_label (urgency_label)
);
