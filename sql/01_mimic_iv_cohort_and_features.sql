/*============================================================================== 
PROJECT: ICU 30-Day Readmission Risk – MIMIC-IV (Production Cohort Build)
AUTHOR: Jyoti Prakash Das
VERSION: 5.1 (Production, cleaned & documented)
LAST UPDATED: 2025-12-09

OVERALL PROJECT GOAL
--------------------
Build a high-quality, production-grade dataset from MIMIC-IV to train and deploy
machine learning models that predict 30-day ICU readmission risk.

This cohort table is the **foundation**:
  - EXACTLY ONE ROW PER PATIENT (first ICU stay only)
  - 30-day ICU readmission label (readmit_30d_flag)
  - Basic demographics & admission details
  - ICU + hospital length of stay and timing metadata

Later feature tables (vitals, labs, GCS, meds, comorbidities, prior history, etc.)
are all LEFT JOINed to THIS table using (subject_id, hadm_id, index_stay_id).

DATA SOURCES (PUBLIC MIMIC-IV)
------------------------------
1) physionet-data.mimiciv_3_1_icu.icustays      → ICU stay timings
2) physionet-data.mimiciv_3_1_hosp.patients     → Demographics
3) physionet-data.mimiciv_3_1_hosp.admissions   → Hospital outcomes

OUTPUT TABLE
------------
nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d

GRAIN
-----
One row per patient:
  - First ICU stay only (index_stay_id)
  - Readmission status within 30 days after ICU discharge

INCLUSION CRITERIA
------------------
✓ Age ≥ 18 years (based on anchor_age)
✓ ICU length of stay ≥ 24 hours
✓ Survived index admission (hospital_expire_flag = 0 or NULL)
✓ Valid admittime & dischtime

EXCLUSION CRITERIA
------------------
✗ Age < 18
✗ ICU LOS < 24 hours
✗ Died in index hospital admission
✗ Invalid/missing admittime or dischtime

READMISSION LABEL (PRIMARY OUTCOME)
-----------------------------------
readmit_30d_flag = 1  → patient had ANY ICU stay > index ICU discharge
                         AND within 1–30 days after index ICU discharge.

readmit_30d_flag = 0  → no ICU stays within that 1–30 day window.

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Correct readmission window:
   - Used all ICU stays for that subject (all_icu_stays) and found the earliest
     ICU intime AFTER index ICU outtime.
   - days_to_next_icu = TIMESTAMP_DIFF(next_intime, index_outtime, DAY).
   - Label = 1 ONLY if 1 ≤ days_to_next_icu ≤ 30.

2) Multiple ICU stays per HADM / per subject:
   - Patients can have multiple ICU stays and multiple admissions.
   - We kept only the **first ICU stay per subject** as index_stay_id using:
     ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY icu_intime).

3) Mortality & LOS filters:
   - Excluded in-hospital deaths in the index admission so readmission is
     well-defined as an outcome.
   - Excluded very short ICU stays (< 24h) to avoid observation-only cases.

OUTPUT COLUMNS (FEATURE DEFINITIONS)
------------------------------------
Each row in the final table has the following columns:

  - subject_id:
      De-identified patient identifier (constant across all admissions/ICU stays).
  - hadm_id:
      Hospital admission ID for the index ICU stay (one hospital visit).
  - index_stay_id:
      ICU stay ID chosen as the INDEX stay (first qualifying ICU stay per subject).
  - index_icu_intime:
      Timestamp when the index ICU stay started (patient entered ICU).
  - index_icu_outtime:
      Timestamp when the index ICU stay ended (patient left ICU).
  - index_icu_los_minutes:
      ICU length of stay for index stay, in minutes.
  - index_icu_los_hours:
      ICU length of stay for index stay, in hours.
  - index_icu_los_days:
      ICU length of stay for index stay, in days (for readability/EDA).
  - gender:
      Recorded sex of the patient at time of data anchor (M/F).
  - anchor_age:
      Approximate age in years at admission (shifted but internally consistent).
  - anchor_year_group:
      Year-group bucket used by MIMIC-IV for date shifting (e.g., "2008 - 2010").
  - admittime:
      Timestamp when the hospital admission (hadm_id) started.
  - dischtime:
      Timestamp when the hospital admission (hadm_id) ended (hospital discharge).
  - admission_type:
      Type of admission (e.g. EMERGENCY, ELECTIVE, OBSERVATION).
  - admission_location:
      Where the patient came from before hospital admission (e.g. ED, clinic).
  - discharge_location:
      Where the patient went after discharge (e.g. HOME, SNF, REHAB).
  - insurance:
      Payer category (e.g. Medicare, Medicaid, Private) – proxy for socioeconomic status.
  - race:
      Race/ethnicity category recorded in the EHR.
  - first_careunit:
      First ICU care unit for the index stay (e.g. MICU, SICU).
  - last_careunit:
      Last ICU care unit for the index stay (if transferred within ICU types).
  - hospital_los_days:
      Total length of stay for the whole hospital admission, in days.
  - mortality_in_index_admission:
      1 if the patient died during the index hospital admission, 0 otherwise.
  - readmit_30d_flag:
      Main ML label: 1 if patient had ANY ICU readmission 1–30 days after index ICU
      discharge, otherwise 0.
  - days_to_30d_readmission:
      Number of days between index ICU discharge and the first ICU readmission
      (only filled if readmit_30d_flag = 1 and within 30 days, else NULL).
  - next_icu_intime_after_index:
      Timestamp of the NEXT ICU admission start after index ICU discharge
      (can be beyond 30 days; NULL if no subsequent ICU stay).
  - cohort_creation_timestamp:
      Timestamp when this cohort row was created (for reproducibility/versioning).

==============================================================================*/

CREATE OR REPLACE TABLE
  `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d` AS

-- ============================================================================
-- STEP 1: Raw ICU stays with LOS metrics
-- ============================================================================
WITH icu_raw AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    intime,
    outtime,
    first_careunit,
    last_careunit,
    TIMESTAMP_DIFF(outtime, intime, MINUTE) AS icu_los_minutes,
    TIMESTAMP_DIFF(outtime, intime, HOUR)   AS icu_los_hours,
    TIMESTAMP_DIFF(outtime, intime, DAY)    AS icu_los_days
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE intime IS NOT NULL
    AND outtime IS NOT NULL
    AND intime < outtime
),

-- ============================================================================
-- STEP 2: Patient demographics (anchor age/year group)
-- ============================================================================
patient_demographics AS (
  SELECT
    subject_id,
    gender,
    anchor_age,
    anchor_year_group
  FROM `physionet-data.mimiciv_3_1_hosp.patients`
  WHERE subject_id IS NOT NULL
),

-- ============================================================================
-- STEP 3: Hospital admissions (LOS + discharge outcomes)
-- ============================================================================
admission_data AS (
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    deathtime,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    race,
    hospital_expire_flag,
    TIMESTAMP_DIFF(dischtime, admittime, DAY) AS hospital_los_days
  FROM `physionet-data.mimiciv_3_1_hosp.admissions`
  WHERE hadm_id IS NOT NULL
    AND admittime IS NOT NULL
    AND dischtime IS NOT NULL
),

-- ============================================================================
-- STEP 4: Join ICU stays with demographics + admissions
-- ============================================================================
icu_with_demographics AS (
  SELECT
    icu.subject_id,
    icu.hadm_id,
    icu.stay_id,
    icu.intime        AS icu_intime,
    icu.outtime       AS icu_outtime,
    icu.icu_los_minutes,
    icu.icu_los_hours,
    icu.icu_los_days,
    icu.first_careunit,
    icu.last_careunit,
    dem.gender,
    dem.anchor_age,
    dem.anchor_year_group,
    adm.admittime,
    adm.dischtime,
    adm.deathtime,
    adm.admission_type,
    adm.admission_location,
    adm.discharge_location,
    adm.insurance,
    adm.race,
    adm.hospital_expire_flag,
    adm.hospital_los_days
  FROM icu_raw icu
  LEFT JOIN patient_demographics dem
    ON icu.subject_id = dem.subject_id
  LEFT JOIN admission_data adm
    ON icu.hadm_id = adm.hadm_id
),

-- ============================================================================
-- STEP 5: Apply cohort filters (age, LOS, survival)
-- ============================================================================
cohort_filtered AS (
  SELECT
    *,
    -- convenience flag for in-hospital mortality
    CASE WHEN deathtime IS NOT NULL THEN 1 ELSE 0 END AS mortality_in_index_admission
  FROM icu_with_demographics
  WHERE SAFE_CAST(anchor_age AS INT64) >= 18      -- Adult patients only
    AND icu_los_minutes >= 1440                   -- ICU LOS ≥ 24 hours
    AND (hospital_expire_flag IS NULL
         OR hospital_expire_flag = 0)             -- survived index admission
    AND admittime IS NOT NULL
    AND dischtime IS NOT NULL
),

-- ============================================================================
-- STEP 6: Select FIRST ICU stay per patient (index stay)
-- ============================================================================
first_icu_per_patient AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY subject_id
      ORDER BY icu_intime ASC
    ) AS rn_icu_order
  FROM cohort_filtered
),

index_stays AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id             AS index_stay_id,
    icu_intime          AS index_icu_intime,
    icu_outtime         AS index_icu_outtime,
    icu_los_minutes     AS index_icu_los_minutes,
    icu_los_hours       AS index_icu_los_hours,
    icu_los_days        AS index_icu_los_days,
    gender,
    anchor_age,
    anchor_year_group,
    admittime,
    dischtime,
    deathtime,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    race,
    first_careunit,
    last_careunit,
    hospital_expire_flag,
    hospital_los_days,
    mortality_in_index_admission
  FROM first_icu_per_patient
  WHERE rn_icu_order = 1      -- keep only first ICU stay per subject
),

-- ============================================================================
-- STEP 7: Get ALL ICU stays (for readmission search)
-- ============================================================================
all_icu_stays AS (
  SELECT
    subject_id,
    stay_id,
    hadm_id,
    intime,
    outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE intime IS NOT NULL
    AND outtime IS NOT NULL
),

-- ============================================================================
-- STEP 8: For each index stay, find earliest ICU stay AFTER index discharge
-- ============================================================================
next_readmission_lookup AS (
  SELECT
    idx.subject_id,
    idx.index_stay_id,
    idx.index_icu_outtime,
    MIN(all_stays.intime) AS next_icu_intime_after_index
  FROM index_stays idx
  LEFT JOIN all_icu_stays all_stays
    ON idx.subject_id = all_stays.subject_id
    -- only consider ICU stays that start AFTER index ICU discharge
    AND all_stays.intime > idx.index_icu_outtime
  GROUP BY
    idx.subject_id,
    idx.index_stay_id,
    idx.index_icu_outtime
),

-- ============================================================================
-- STEP 9: Attach readmission timing (days_to_next_icu)
-- ============================================================================
index_with_readmission AS (
  SELECT
    idx.*,
    nrl.next_icu_intime_after_index,
    TIMESTAMP_DIFF(
      nrl.next_icu_intime_after_index,
      idx.index_icu_outtime,
      DAY
    ) AS days_to_next_icu
  FROM index_stays idx
  LEFT JOIN next_readmission_lookup nrl
    ON idx.subject_id    = nrl.subject_id
   AND idx.index_stay_id = nrl.index_stay_id
),

-- ============================================================================
-- STEP 10: Compute 30-day readmission label
-- ============================================================================
final_cohort AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    index_icu_intime,
    index_icu_outtime,
    index_icu_los_minutes,
    index_icu_los_hours,
    index_icu_los_days,
    gender,
    anchor_age,
    anchor_year_group,
    admittime,
    dischtime,
    deathtime,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    race,
    first_careunit,
    last_careunit,
    hospital_expire_flag,
    hospital_los_days,
    mortality_in_index_admission,
    next_icu_intime_after_index,
    days_to_next_icu,

    -- PRIMARY OUTCOME: 30-day ICU readmission flag
    CASE
      WHEN next_icu_intime_after_index IS NOT NULL
       AND days_to_next_icu BETWEEN 1 AND 30
      THEN 1
      ELSE 0
    END AS readmit_30d_flag,

    -- Secondary outcome: days to ICU readmission (only if within 30 days)
    CASE
      WHEN next_icu_intime_after_index IS NOT NULL
       AND days_to_next_icu BETWEEN 1 AND 30
      THEN days_to_next_icu
      ELSE NULL
    END AS days_to_30d_readmission
  FROM index_with_readmission
)

-- ============================================================================
-- STEP 11: Final SELECT (one row per patient)
-- ============================================================================
SELECT
  subject_id,
  hadm_id,
  index_stay_id,
  index_icu_intime,
  index_icu_outtime,
  index_icu_los_minutes,
  index_icu_los_hours,
  index_icu_los_days,
  gender,
  anchor_age,
  anchor_year_group,
  admittime,
  dischtime,
  admission_type,
  admission_location,
  discharge_location,
  insurance,
  race,
  first_careunit,
  last_careunit,
  hospital_los_days,
  mortality_in_index_admission,
  -- OUTCOMES
  readmit_30d_flag,
  days_to_30d_readmission,
  next_icu_intime_after_index,
  -- METADATA
  CURRENT_TIMESTAMP() AS cohort_creation_timestamp
FROM final_cohort
ORDER BY subject_id ASC;


-- ============================================================================
-- AUDIT 1: Cohort size & high-level stats
-- ============================================================================
SELECT
  COUNT(*) AS total_patients_in_cohort,
  COUNT(DISTINCT subject_id) AS unique_patients,
  SUM(readmit_30d_flag) AS n_readmitted_30d,
  ROUND(100.0 * SUM(readmit_30d_flag) / COUNT(*), 2) AS pct_readmitted_30d,
  SUM(mortality_in_index_admission) AS n_mortality_index,
  ROUND(100.0 * SUM(mortality_in_index_admission) / COUNT(*), 2) AS pct_mortality_index,
  ROUND(AVG(anchor_age), 1) AS mean_age,
  ROUND(AVG(index_icu_los_days), 1) AS mean_icu_los_days,
  ROUND(AVG(hospital_los_days), 1) AS mean_hospital_los_days
FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`;

-- ============================================================================
-- AUDIT 2: Inspect first 100 readmitted cases
-- ============================================================================
SELECT
  subject_id,
  hadm_id,
  index_stay_id,
  index_icu_intime,
  index_icu_outtime,
  next_icu_intime_after_index,
  days_to_30d_readmission,
  gender,
  anchor_age,
  hospital_los_days
FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`
WHERE readmit_30d_flag = 1
ORDER BY days_to_30d_readmission ASC
LIMIT 100;

-- ============================================================================
-- AUDIT 3: Distribution of days to readmission (among readmitted)
-- ============================================================================
SELECT
  days_to_30d_readmission,
  COUNT(*) AS n_cases,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_readmissions
FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`
WHERE readmit_30d_flag = 1
GROUP BY days_to_30d_readmission
ORDER BY days_to_30d_readmission;

-- ============================================================================
-- AUDIT 4: Demographics & readmission by gender
-- ============================================================================
SELECT
  gender,
  COUNT(*) AS n_patients,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
  ROUND(AVG(anchor_age), 1) AS mean_age,
  MIN(anchor_age) AS min_age,
  MAX(anchor_age) AS max_age,
  SUM(readmit_30d_flag) AS n_readmitted,
  ROUND(100.0 * SUM(readmit_30d_flag) / COUNT(*), 2) AS readmit_pct
FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`
GROUP BY gender
ORDER BY n_patients DESC;


-- ok physionet have already charlseon table score pre audited and given , so will look what it all provides, as we will be using thoer table as well as cutom made table from our side as well

-- (A) Schema: list columns for physionet derived charlson
SELECT column_name, data_type, ordinal_position
FROM `physionet-data.mimiciv_3_1_derived.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'charlson'
ORDER BY ordinal_position;

-- (B) Quick preview (restricted to our cohort for fast preview)
SELECT *
FROM `physionet-data.mimiciv_3_1_derived.charlson`
WHERE hadm_id IN (SELECT hadm_id FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`)
LIMIT 10;

-- Schema for our local hand-built charlson_score
SELECT column_name, data_type, ordinal_position
FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'charlson_score'
ORDER BY ordinal_position;

-- Schema for mapping flags
SELECT column_name, data_type, ordinal_position
FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'feature_comorbidities_mapping_flags'
ORDER BY ordinal_position;

-- Quick preview rows (to inspect values)
SELECT * FROM `nomadic-freedom-436306-g4.readmission30.charlson_score` LIMIT 5;
SELECT * FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags` LIMIT 5;



/* =============================================================================
PROJECT: ICU 30-Day Readmission – Charlson (Merged: PhysioNet ∨ Local ∨ Mapping)
AUTHOR : Jyoti Prakash Das
VERSION: 1.0 (Merged, cohort-restricted)
LAST UPDATED: 2025-12-09

PURPOSE
-------
This table was built to provide a single, auditable Charlson-style comorbidity
source for our ML pipeline. For every admission in our cohort we:
  1) Preferred the audited PhysioNet-derived Charlson row if present.
  2) Otherwise fell back to our local cohort-restricted charlson_score.
  3) If both were missing, computed a Charlson-like weighted index from our
     mapping-derived flags (feature_comorbidities_mapping_flags).

This ensured maximum coverage, reproducibility and suitability for production
model training and deployment.

DATA SOURCES (used)
-------------------
- nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d  (cohort keys)
- physionet-data.mimiciv_3_1_derived.charlson                      (audited / preferred)
- nomadic-freedom-436306-g4.readmission30.charlson_score           (local computed)
- nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags (mapping flags)

OUTPUT TABLE
------------
nomadic-freedom-436306-g4.readmission30.charlson_merged
Grain: one row per (subject_id, hadm_id) in our cohort.

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Cohort-restricted merging — only hadm_ids in our cohort were scanned.
2) Preference order — PhysioNet (audited) → Local (reproducible) → Mapping fallback.
3) Types & safety — used SAFE_CAST and explicit numeric CASEs to avoid BOOL/INT errors.
4) Traceability — included a charlson_source column to record which input was used.
5) Reproducibility — final index was computed deterministically from flags where necessary.

USAGE
-----
- This table was joined into downstream final_model_dataset / feature joins
  using (subject_id, hadm_id).
- The charlson_index_final was used as a numeric predictor; flags were used as
  interpretable features.

/* ─────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions + Why Useful for ICU Readmission Prediction)
───────────────────────────────────────────────────────────────────────────────

Primary Keys
• subject_id  
      → Unique patient identifier. Used for patient-level joins.
• hadm_id  
      → Unique hospital admission identifier. Core join key to other feature tables.

Source Tracking
• charlson_source  
      → Shows where the comorbidity score was taken from:
         'physionet' = validated score from official MIMIC derived table,
         'local'     = re-computed via ICD mapping (if physionet missing),
         'mapping_fallback' = last-resort ICD flag-based computation,
         'none'      = no comorbidity evidence.
      → Helps in debugging, trust scoring, and research reproducibility.

Binary Clinical Comorbidity Features (0 = condition absent, 1 = present)
These represent chronic conditions strongly associated with readmission and mortality.

• myocardial_infarct  
      → History of heart attack. Increases cardiac risk → higher readmission probability.
• congestive_heart_failure  
      → Chronic heart failure. Often leads to fluid overload & recurrent admissions.
• peripheral_vascular_disease  
      → Poor arterial circulation. Linked to complications & re-hospitalizations.
• cerebrovascular_disease  
      → Stroke/TIA history. Higher neurological impairment → slower recovery.
• dementia  
      → Cognitive impairment → medication non-compliance & caregiver dependency.
• chronic_pulmonary_disease  
      → COPD / chronic lung disease. High risk of exacerbations → readmissions.
• rheumatic_disease  
      → Autoimmune connective tissue disease → chronic inflammation & flare-ups.
• peptic_ulcer_disease  
      → GI complications. Risk of bleeding → hospital return.
• mild_liver_disease  
      → Early hepatic dysfunction. Predictive for long-term complications.
• diabetes_without_cc  
      → Diabetes (controlled). Still increases infection & wound risk.
• diabetes_with_cc  
      → Diabetes with complications (retinopathy/nephropathy). Strong readmission risk.
• paraplegia  
      → Paralysis. High dependency & infection risk → return likelihood increases.
• renal_disease  
      → CKD / ESRD. Strongest driver of readmission due to dialysis burden.
• malignant_cancer  
      → Solid tumor without metastasis. Readmission often due to chemo/complications.
• severe_liver_disease  
      → Advanced cirrhosis. Very high mortality + readmission probability.
• metastatic_solid_tumor  
      → Cancer having spread. Extremely important severity indicator.
• aids  
      → HIV/AIDS. Immunosuppression increases infection-related readmissions.

Final Risk Score Features
• charlson_index_final  
      → Final comorbidity burden used in ML model (INT).  
         Higher = sicker patient = increased readmission risk.
• pn_charlson_index  
      → Score from official PhysioNet derived table (gold-standard reference).
• local_charlson_index  
      → Score computed by our ICD9/ICD10 rules. Used only when PN unavailable.
• mapping_charlson_index  
      → Fallback aggregated score from mapping flags.
      → These three ensure robustness for real-world hospitals lacking PN data.

Metadata
• feature_extraction_ts  
      → Timestamp of creation. Useful for reproducibility & version auditing.

──────────────────────────────────────────────────────────────────────────────*/


CREATE OR REPLACE TABLE
  `nomadic-freedom-436306-g4.readmission30.charlson_merged` AS

WITH cohort AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`
),

-- 1) PhysioNet (preferred) — restricted to our cohort for speed
physionet AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(age_score AS INT64) AS pn_age_score,
    SAFE_CAST(myocardial_infarct AS INT64) AS pn_myocardial_infarct,
    SAFE_CAST(congestive_heart_failure AS INT64) AS pn_congestive_heart_failure,
    SAFE_CAST(peripheral_vascular_disease AS INT64) AS pn_peripheral_vascular_disease,
    SAFE_CAST(cerebrovascular_disease AS INT64) AS pn_cerebrovascular_disease,
    SAFE_CAST(dementia AS INT64) AS pn_dementia,
    SAFE_CAST(chronic_pulmonary_disease AS INT64) AS pn_chronic_pulmonary_disease,
    SAFE_CAST(rheumatic_disease AS INT64) AS pn_rheumatic_disease,
    SAFE_CAST(peptic_ulcer_disease AS INT64) AS pn_peptic_ulcer_disease,
    SAFE_CAST(mild_liver_disease AS INT64) AS pn_mild_liver_disease,
    SAFE_CAST(diabetes_without_cc AS INT64) AS pn_diabetes_without_cc,
    SAFE_CAST(diabetes_with_cc AS INT64) AS pn_diabetes_with_cc,
    SAFE_CAST(paraplegia AS INT64) AS pn_paraplegia,
    SAFE_CAST(renal_disease AS INT64) AS pn_renal_disease,
    SAFE_CAST(malignant_cancer AS INT64) AS pn_malignant_cancer,
    SAFE_CAST(severe_liver_disease AS INT64) AS pn_severe_liver_disease,
    SAFE_CAST(metastatic_solid_tumor AS INT64) AS pn_metastatic_solid_tumor,
    SAFE_CAST(aids AS INT64) AS pn_aids,
    SAFE_CAST(charlson_comorbidity_index AS INT64) AS pn_charlson_index
  FROM `physionet-data.mimiciv_3_1_derived.charlson`
  WHERE hadm_id IN (SELECT hadm_id FROM cohort)
),

-- 2) Our local computed charlson_score (cohort-restricted)
local AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(myocardial_infarct AS INT64) AS lc_myocardial_infarct,
    SAFE_CAST(congestive_heart_failure AS INT64) AS lc_congestive_heart_failure,
    SAFE_CAST(peripheral_vascular_disease AS INT64) AS lc_peripheral_vascular_disease,
    SAFE_CAST(cerebrovascular_disease AS INT64) AS lc_cerebrovascular_disease,
    SAFE_CAST(dementia AS INT64) AS lc_dementia,
    SAFE_CAST(chronic_pulmonary_disease AS INT64) AS lc_chronic_pulmonary_disease,
    SAFE_CAST(rheumatic_disease AS INT64) AS lc_rheumatic_disease,
    SAFE_CAST(peptic_ulcer_disease AS INT64) AS lc_peptic_ulcer_disease,
    SAFE_CAST(mild_liver_disease AS INT64) AS lc_mild_liver_disease,
    SAFE_CAST(diabetes_without_cc AS INT64) AS lc_diabetes_without_cc,
    SAFE_CAST(diabetes_with_cc AS INT64) AS lc_diabetes_with_cc,
    SAFE_CAST(paraplegia AS INT64) AS lc_paraplegia,
    SAFE_CAST(renal_disease AS INT64) AS lc_renal_disease,
    SAFE_CAST(malignant_cancer AS INT64) AS lc_malignant_cancer,
    SAFE_CAST(severe_liver_disease AS INT64) AS lc_severe_liver_disease,
    SAFE_CAST(metastatic_solid_tumor AS INT64) AS lc_metastatic_solid_tumor,
    SAFE_CAST(aids AS INT64) AS lc_aids,
    SAFE_CAST(charlson_comorbidity_index AS INT64) AS lc_charlson_index
  FROM `nomadic-freedom-436306-g4.readmission30.charlson_score`
  WHERE hadm_id IN (SELECT hadm_id FROM cohort)
),

-- 3) Mapping flags (used for fallback / extra coverage)
mapping AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(myocardial_infarction_flag AS INT64)  AS map_myocardial_infarct,
    SAFE_CAST(chf_flag AS INT64)                    AS map_congestive_heart_failure,
    SAFE_CAST(peripheral_vascular_flag AS INT64)    AS map_peripheral_vascular_disease,
    SAFE_CAST(cerebrovascular_flag AS INT64)        AS map_cerebrovascular_disease,
    SAFE_CAST(dementia_flag AS INT64)               AS map_dementia,
    SAFE_CAST(copd_flag AS INT64)                   AS map_chronic_pulmonary_disease,
    SAFE_CAST(NULLIF(diabetes_no_cc_flag,0) AS INT64)    AS map_diabetes_without_cc,
    SAFE_CAST(diabetes_with_cc_flag AS INT64)       AS map_diabetes_with_cc,
    SAFE_CAST(ckd_flag AS INT64)                    AS map_renal_disease,
    SAFE_CAST(cancer_flag AS INT64)                 AS map_malignant_cancer,
    SAFE_CAST(metastatic_tumor_flag AS INT64)       AS map_metastatic_solid_tumor,
    SAFE_CAST(mild_liver_flag AS INT64)             AS map_mild_liver_disease,
    SAFE_CAST(severe_liver_flag AS INT64)           AS map_severe_liver_disease,
    SAFE_CAST(paraplegia_flag AS INT64)             AS map_paraplegia,
    SAFE_CAST(aids_flag AS INT64)                   AS map_aids,
    SAFE_CAST(afib_flag AS INT64)                   AS map_afib_flag,
    SAFE_CAST(hypertension_flag AS INT64)           AS map_hypertension_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags`
  WHERE hadm_id IN (SELECT hadm_id FROM cohort)
),

-- 4) Compute a Charlson-like weighted index from mapping flags (if needed)
mapping_index AS (
  SELECT
    m.subject_id,
    m.hadm_id,
    -- apply classical Charlson weights to mapping flags where available
    (
      COALESCE(m.map_myocardial_infarct,0) * 1
    + COALESCE(m.map_congestive_heart_failure,0) * 1
    + COALESCE(m.map_peripheral_vascular_disease,0) * 1
    + COALESCE(m.map_cerebrovascular_disease,0) * 1
    + COALESCE(m.map_dementia,0) * 1
    + COALESCE(m.map_chronic_pulmonary_disease,0) * 1
    + 0 /* rheumatic not present in mapping_flags, assumed 0 */
    + 0 /* peptic_ulcer not present in mapping_flags, assumed 0 */
    + COALESCE(m.map_mild_liver_disease,0) * 1
    + COALESCE(m.map_diabetes_without_cc,0) * 1
    + COALESCE(m.map_diabetes_with_cc,0) * 2
    + COALESCE(m.map_paraplegia,0) * 2
    + COALESCE(m.map_renal_disease,0) * 2
    + COALESCE(m.map_malignant_cancer,0) * 2
    + COALESCE(m.map_severe_liver_disease,0) * 3
    + COALESCE(m.map_metastatic_solid_tumor,0) * 6
    + COALESCE(m.map_aids,0) * 6
    ) AS mapping_charlson_index
  FROM mapping m
)

-- Final assembly: choose best available source (physionet → local → mapping)
SELECT
  c.subject_id,
  c.hadm_id,

  -- chosen source label
  CASE
    WHEN pn.hadm_id IS NOT NULL THEN 'physionet'
    WHEN lc.hadm_id IS NOT NULL THEN 'local'
    WHEN mi.hadm_id IS NOT NULL THEN 'mapping_fallback'
    ELSE 'none'
  END AS charlson_source,

  -- Binary flags: prefer physionet → local → mapping flag (0/1)
  COALESCE(pn.pn_myocardial_infarct, lc.lc_myocardial_infarct,
           CASE WHEN mapping.map_myocardial_infarct = 1 THEN 1 ELSE 0 END, 0)
    AS myocardial_infarct,
  COALESCE(pn.pn_congestive_heart_failure, lc.lc_congestive_heart_failure,
           CASE WHEN mapping.map_congestive_heart_failure = 1 THEN 1 ELSE 0 END, 0)
    AS congestive_heart_failure,
  COALESCE(pn.pn_peripheral_vascular_disease, lc.lc_peripheral_vascular_disease,
           CASE WHEN mapping.map_peripheral_vascular_disease = 1 THEN 1 ELSE 0 END, 0)
    AS peripheral_vascular_disease,
  COALESCE(pn.pn_cerebrovascular_disease, lc.lc_cerebrovascular_disease,
           CASE WHEN mapping.map_cerebrovascular_disease = 1 THEN 1 ELSE 0 END, 0)
    AS cerebrovascular_disease,
  COALESCE(pn.pn_dementia, lc.lc_dementia,
           CASE WHEN mapping.map_dementia = 1 THEN 1 ELSE 0 END, 0)
    AS dementia,
  COALESCE(pn.pn_chronic_pulmonary_disease, lc.lc_chronic_pulmonary_disease,
           CASE WHEN mapping.map_chronic_pulmonary_disease = 1 THEN 1 ELSE 0 END, 0)
    AS chronic_pulmonary_disease,
  COALESCE(pn.pn_rheumatic_disease, lc.lc_rheumatic_disease, 0) AS rheumatic_disease,
  COALESCE(pn.pn_peptic_ulcer_disease, lc.lc_peptic_ulcer_disease, 0) AS peptic_ulcer_disease,
  COALESCE(pn.pn_mild_liver_disease, lc.lc_mild_liver_disease,
           CASE WHEN mapping.map_mild_liver_disease = 1 THEN 1 ELSE 0 END, 0)
    AS mild_liver_disease,
  COALESCE(pn.pn_diabetes_without_cc, lc.lc_diabetes_without_cc,
           CASE WHEN mapping.map_diabetes_without_cc = 1 THEN 1 ELSE 0 END, 0)
    AS diabetes_without_cc,
  COALESCE(pn.pn_diabetes_with_cc, lc.lc_diabetes_with_cc,
           CASE WHEN mapping.map_diabetes_with_cc = 1 THEN 1 ELSE 0 END, 0)
    AS diabetes_with_cc,
  COALESCE(pn.pn_paraplegia, lc.lc_paraplegia,
           CASE WHEN mapping.map_paraplegia = 1 THEN 1 ELSE 0 END, 0)
    AS paraplegia,
  COALESCE(pn.pn_renal_disease, lc.lc_renal_disease,
           CASE WHEN mapping.map_renal_disease = 1 THEN 1 ELSE 0 END, 0)
    AS renal_disease,
  COALESCE(pn.pn_malignant_cancer, lc.lc_malignant_cancer,
           CASE WHEN mapping.map_malignant_cancer = 1 THEN 1 ELSE 0 END, 0)
    AS malignant_cancer,
  COALESCE(pn.pn_severe_liver_disease, lc.lc_severe_liver_disease,
           CASE WHEN mapping.map_severe_liver_disease = 1 THEN 1 ELSE 0 END, 0)
    AS severe_liver_disease,
  COALESCE(pn.pn_metastatic_solid_tumor, lc.lc_metastatic_solid_tumor,
           CASE WHEN mapping.map_metastatic_solid_tumor = 1 THEN 1 ELSE 0 END, 0)
    AS metastatic_solid_tumor,
  COALESCE(pn.pn_aids, lc.lc_aids,
           CASE WHEN mapping.map_aids = 1 THEN 1 ELSE 0 END, 0)
    AS aids,

  -- final charlson index: prefer physionet -> local -> mapping-computed
  COALESCE(pn.pn_charlson_index, lc.lc_charlson_index, mi.mapping_charlson_index, 0) AS charlson_index_final,

  -- provenance values for debugging/audit
  pn.pn_charlson_index AS pn_charlson_index,
  lc.lc_charlson_index AS local_charlson_index,
  mi.mapping_charlson_index AS mapping_charlson_index,

  CURRENT_TIMESTAMP() AS feature_extraction_ts

FROM cohort c
LEFT JOIN physionet pn ON c.hadm_id = pn.hadm_id
LEFT JOIN local lc       ON c.hadm_id = lc.hadm_id
LEFT JOIN mapping       ON c.hadm_id = mapping.hadm_id
LEFT JOIN mapping_index mi ON c.hadm_id = mi.hadm_id

ORDER BY subject_id, hadm_id;


/* ============================================================================
QC BLOCK: charlson_merged  (Coverage, Consistency, Agreement, Sanity Checks)
TABLE  : nomadic-freedom-436306-g4.readmission30.charlson_merged
GOAL   : Make sure our merged Charlson table behaves as expected before modeling
============================================================================ */


/* ---------------------------------------------------------------------------
QC 1: Overall coverage & source distribution
- What it checked:
    • Did we get exactly one row per cohort admission?
    • From which source did each row come? (physionet / local / mapping_fallback / none)
- Why it mattered:
    • We wanted most rows to come from physionet if available.
    • We wanted to confirm that the fallback logic actually filled gaps.
--------------------------------------------------------------------------- */
SELECT
  COUNT(*) AS total_rows_in_charlson_merged,
  COUNT(DISTINCT subject_id) AS unique_subjects,
  COUNT(DISTINCT hadm_id)    AS unique_hadm_ids,

  -- counts by source
  SUM(CASE WHEN charlson_source = 'physionet'         THEN 1 ELSE 0 END) AS n_physionet,
  SUM(CASE WHEN charlson_source = 'local'             THEN 1 ELSE 0 END) AS n_local,
  SUM(CASE WHEN charlson_source = 'mapping_fallback'  THEN 1 ELSE 0 END) AS n_mapping_fallback,
  SUM(CASE WHEN charlson_source = 'none'              THEN 1 ELSE 0 END) AS n_none,

  -- percentages by source
  ROUND(100.0 * SUM(CASE WHEN charlson_source = 'physionet'        THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS pct_physionet,
  ROUND(100.0 * SUM(CASE WHEN charlson_source = 'local'            THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS pct_local,
  ROUND(100.0 * SUM(CASE WHEN charlson_source = 'mapping_fallback' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS pct_mapping_fallback,
  ROUND(100.0 * SUM(CASE WHEN charlson_source = 'none'             THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS pct_none
FROM `nomadic-freedom-436306-g4.readmission30.charlson_merged`;


/* ---------------------------------------------------------------------------
QC 2: Sanity check – uniqueness of (subject_id, hadm_id)
- What it checked:
    • There should be exactly ONE row per admission in our cohort.
- Why it mattered:
    • Duplicates would break joins and inflate counts in modeling.
--------------------------------------------------------------------------- */
SELECT
  hadm_id,
  COUNT(*) AS cnt
FROM `nomadic-freedom-436306-g4.readmission30.charlson_merged`
GROUP BY hadm_id
HAVING cnt > 1
LIMIT 20;   -- should return 0 rows; any row here = a bug to investigate


/* ---------------------------------------------------------------------------
QC 3: Basic distribution of Charlson scores
- What it checked:
    • Overall spread of charlson_index_final.
    • Rough shape (min / max / quartiles) and how many are zero vs non-zero.
- Why it mattered:
    • We wanted to confirm there was variation (not all zeros),
      and that the range looked clinically reasonable.
--------------------------------------------------------------------------- */
SELECT
  COUNT(*) AS n_rows,
  MIN(charlson_index_final) AS min_charlson,
  MAX(charlson_index_final) AS max_charlson,
  ROUND(AVG(charlson_index_final), 2) AS mean_charlson,
  -- simple percentile approximations via APPROX_QUANTILES
  APPROX_QUANTILES(charlson_index_final, 5)[OFFSET(1)] AS q25_charlson,
  APPROX_QUANTILES(charlson_index_final, 5)[OFFSET(2)] AS q50_charlson,
  APPROX_QUANTILES(charlson_index_final, 5)[OFFSET(3)] AS q75_charlson,
  SUM(CASE WHEN charlson_index_final = 0 THEN 1 ELSE 0 END) AS n_zero_score,
  ROUND(100.0 * SUM(CASE WHEN charlson_index_final = 0 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS pct_zero_score
FROM `nomadic-freedom-436306-g4.readmission30.charlson_merged`;


/* ---------------------------------------------------------------------------
QC 4: Agreement with PhysioNet Charlson when PhysioNet was available
- What it checked:
    • Among admissions where PhysioNet gave a Charlson index,
      how often did our charlson_index_final match it exactly?
    • What was the mean absolute difference when it did not match?
- Why it mattered:
    • This told us whether merging + fallback logic preserved audited values
      or changed them unexpectedly.
--------------------------------------------------------------------------- */
SELECT
  COUNT(*) AS n_with_physionet,
  SUM(CASE
        WHEN pn_charlson_index IS NOT NULL
         AND SAFE_CAST(pn_charlson_index AS INT64) = SAFE_CAST(charlson_index_final AS INT64)
        THEN 1 ELSE 0
      END) AS n_exact_match,
  ROUND(
    100.0 * SUM(CASE
                  WHEN pn_charlson_index IS NOT NULL
                   AND SAFE_CAST(pn_charlson_index AS INT64) = SAFE_CAST(charlson_index_final AS INT64)
                  THEN 1 ELSE 0
                END)
    / NULLIF(COUNT(*),0), 2
  ) AS pct_exact_match,
  ROUND(
    AVG(
      ABS(
        SAFE_CAST(pn_charlson_index AS INT64)
        - SAFE_CAST(charlson_index_final AS INT64)
      )
    ), 2
  ) AS mean_abs_diff
FROM `nomadic-freedom-436306-g4.readmission30.charlson_merged`
WHERE pn_charlson_index IS NOT NULL;


/* ---------------------------------------------------------------------------
QC 5: Where did mapping_fallback actually contribute?
- What it checked:
    • Which rows had charlson_source = 'mapping_fallback'?
    • What scores did mapping_charlson_index produce vs physionet/local (which
      would be NULL for those rows)?
- Why it mattered:
    • We wanted to see if fallback rows looked clinically reasonable and if we
      were not accidentally assigning crazy high scores.
--------------------------------------------------------------------------- */
SELECT
  charlson_source,
  COUNT(*) AS n_rows,
  MIN(mapping_charlson_index) AS min_mapping_index,
  MAX(mapping_charlson_index) AS max_mapping_index,
  ROUND(AVG(mapping_charlson_index), 2) AS mean_mapping_index
FROM `nomadic-freedom-436306-g4.readmission30.charlson_merged`
WHERE charlson_source = 'mapping_fallback'
GROUP BY charlson_source;


/* ---------------------------------------------------------------------------
QC 6 (Optional but very useful): Relationship to 30-day readmission label
- What it checked:
    • How Charlson score behaved by readmission outcome.
- Why it mattered:
    • Confirmed that higher Charlson burden was associated with higher
      readmission rate (sanity check for model usefulness).
--------------------------------------------------------------------------- */
SELECT
  r.readmit_30d_flag,
  COUNT(*) AS n_rows,
  ROUND(AVG(c.charlson_index_final), 2) AS mean_charlson,
  MIN(c.charlson_index_final) AS min_charlson,
  MAX(c.charlson_index_final) AS max_charlson
FROM `nomadic-freedom-436306-g4.readmission30.charlson_merged` c
JOIN `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d` r
  USING (subject_id, hadm_id)
GROUP BY r.readmit_30d_flag
ORDER BY r.readmit_30d_flag;



/*==============================================================================
PROJECT : ICU 30-Day Readmission – Demographics & Admission Feature Layer
TABLE  : feature_demographics_extended
AUTHOR : Jyoti Prakash Das
VERSION: 1.0 (Cohort-driven, production-ready)
LAST UPDATED: 2025-12-09

PURPOSE
-------
We built this table as the **core demographic & admission feature layer** on top
of our index cohort (mimiciv_index_cohort_30d). It provides:

  - Cleaned patient demographics (age, gender, race)
  - Admission, discharge, insurance, and care unit context
  - Hospital LOS & mortality flag
  - 30-day ICU readmission label + basic timing
  - Simple categorical features (age bands, LOS categories)

Almost every downstream model / feature table joins back to this table via
(subject_id, hadm_id, index_stay_id).

DATA SOURCE
-----------
- nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d

GRAIN
-----
One row per patient’s **index ICU stay** (first ICU stay per subject).

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Single, stable grain
   - We forced one row per (subject_id, hadm_id, index_stay_id), matching the
     cohort. All later feature tables safely LEFT JOIN on this.

2) Age handling
   - We used anchor_age from MIMIC and cast it to INT (age_at_admission).
   - We created age_group bands (18–40, 40–60, 60–80, 80+) used frequently in
     exploratory analysis and baseline models.

3) Readmission label passthrough
   - We carried over readmit_30d_flag and days_to_30d_readmission and made
     them INTs to ensure robustness when exporting to Python/Parquet.

4) LOS & category engineering
   - We kept hospital_los_days as a numeric feature.
   - We derived los_category = Short / Medium / Long / Unknown to capture
     nonlinear relationships between LOS and readmission risk.

5) Metadata for reproducibility
   - We preserved cohort_creation_timestamp and added feature_extraction_timestamp
     so we always know **when** this feature layer was generated.

──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

Core Keys & ICU Index Info
• subject_id
    → Unique patient identifier. Primary key for patient-level joins.
• hadm_id
    → Hospital admission identifier. Key to connect to hosp/derived tables.
• index_stay_id
    → ICU stay ID for the first (index) ICU admission. Used to join to
      ICU-level derived tables (sofa, labs, vitals, meds, etc.).
• index_icu_intime
    → Timestamp of ICU admission for the index stay. Useful for time-based
      alignment with events.
• index_icu_outtime
    → Timestamp of ICU discharge for the index stay. Used as the anchor point
      for 30-day readmission windows.
• index_icu_los_minutes / index_icu_los_hours / index_icu_los_days
    → ICU length of stay (LOS) in different units. Longer LOS often signals
      higher severity / complexity and can correlate with readmission risk.

Demographics
• gender
    → Patient sex recorded in MIMIC. Often associated with different disease
      patterns & healthcare utilization.
• age_at_admission
    → Anchor age (years) at admission. Age is a core risk driver for mortality
      and readmission.
• anchor_year_group
    → Temporal anchor group (e.g., “2008–2010”). Captures changes in practice
      over time, which can influence readmission patterns.

Hospital Admission Details
• admittime
    → Start of the hospital stay. Useful for temporal modelling and checking
      gaps between admissions.
• dischtime
    → End of the hospital stay. Combined with admittime → hospital_los_days.
• admission_type
    → e.g., EMERGENCY, ELECTIVE, URGENT. Emergency admissions tend to be higher
      risk for readmission.
• admission_location
    → Where the patient came from (ED, transfer, clinic). Important for
      understanding baseline acuity and referral patterns.
• discharge_location
    → Where the patient went after discharge (home, rehab, nursing facility).
      Non-home discharges can signal frailty and higher readmission risk.
• insurance
    → Payer type (Medicare, Private, etc.). Proxy for socioeconomic status and
      access to follow-up care.
• race
    → Self-identified race/ethnicity. Can capture structural disparities in
      care and readmission (model must be handled ethically).

Care Unit Context
• first_careunit
    → First ICU location (e.g., MICU, SICU). Different units treat different
      populations (medical vs surgical vs cardiac).
• last_careunit
    → ICU location at discharge. Transfers between units may capture complexity.

LOS & Mortality
• hospital_los_days
    → Total hospital LOS in days. Very short or very long stays both have
      specific readmission patterns.
• mortality_in_index_admission
    → 1 = patient died in the index admission, 0 = survived. For this cohort,
      generally filtered to survivors, but kept for completeness.

Readmission Outcome (Primary Label)
• readmit_30d_flag
    → 1 = ICU readmission within 1–30 days after index ICU discharge;
      0 = no ICU readmission in that window.
      This is the **main target variable** for our prediction model.
• days_to_30d_readmission
    → Number of days between ICU discharge and next ICU admission (if it fell
      in the 1–30 day window). Useful for survival-style modeling and analysis.
• next_icu_intime_after_index
    → Timestamp of the next ICU admission (if any). Good for timeline analysis.

Derived Categorical Features
• age_group
    → Age bucket (18–40, 40–60, 60–80, 80+). Helps simple models / dashboards
      and enables stratified analysis.
• los_category
    → Hospital LOS bucket: Short (<3 days), Medium (3–7), Long (>7), Unknown.
      Provides a discrete view of LOS for readmission risk profiling.

Metadata
• feature_extraction_timestamp
    → Timestamp when this feature table was generated. Critical for data
      lineage and pipeline debugging.
• cohort_creation_timestamp
    → Timestamp when the underlying cohort table was created. Helps track which
      cohort version these features are based on.

==============================================================================
CREATE TABLE: feature_demographics_extended
==============================================================================*/

CREATE OR REPLACE TABLE
  `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended` AS
SELECT
  -- Core identifiers / ICU info
  c.subject_id,
  c.hadm_id,
  c.index_stay_id,
  c.index_icu_intime,
  c.index_icu_outtime,
  c.index_icu_los_minutes,
  c.index_icu_los_hours,
  c.index_icu_los_days,

  -- Demographics
  c.gender,
  SAFE_CAST(c.anchor_age AS INT64) AS age_at_admission,
  c.anchor_year_group,

  -- Hospital admission details
  c.admittime,
  c.dischtime,
  c.admission_type,
  c.admission_location,
  c.discharge_location,
  c.insurance,
  c.race,

  -- Care unit context
  c.first_careunit,
  c.last_careunit,

  -- Hospital stay outcome / LOS
  c.hospital_los_days,
  c.mortality_in_index_admission,

  -- Readmission outcome (30-day)
  SAFE_CAST(c.readmit_30d_flag AS INT64)        AS readmit_30d_flag,
  SAFE_CAST(c.days_to_30d_readmission AS INT64) AS days_to_30d_readmission,
  c.next_icu_intime_after_index,

  -- Derived categorical features
  CASE
    WHEN SAFE_CAST(c.anchor_age AS INT64) < 40 THEN '18-40'
    WHEN SAFE_CAST(c.anchor_age AS INT64) < 60 THEN '40-60'
    WHEN SAFE_CAST(c.anchor_age AS INT64) < 80 THEN '60-80'
    ELSE '80+'
  END AS age_group,

  CASE
    WHEN c.hospital_los_days IS NULL THEN 'Unknown'
    WHEN c.hospital_los_days < 3 THEN 'Short'
    WHEN c.hospital_los_days < 7 THEN 'Medium'
    ELSE 'Long'
  END AS los_category,

  -- Metadata
  CURRENT_TIMESTAMP()          AS feature_extraction_timestamp,
  c.cohort_creation_timestamp

FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d` c
WHERE c.subject_id IS NOT NULL
  AND c.hadm_id IS NOT NULL;





/*==============================================================================
QC BLOCK – feature_demographics_extended
We wrote these checks so we can quickly confirm completeness, plausibility,
and the relationship between demographics and readmission.
==============================================================================*/

-- ============================================================================
-- QC 1: Missing Data Assessment
--   Goal: Ensure core demographic & LOS fields are fully populated.
-- ============================================================================

SELECT
  'QC1_Missing_Data' AS check_name,
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN gender IS NULL THEN 1 END)            AS missing_gender,
  COUNT(CASE WHEN age_at_admission IS NULL THEN 1 END)  AS missing_age,
  COUNT(CASE WHEN admission_type IS NULL THEN 1 END)    AS missing_admission_type,
  COUNT(CASE WHEN hospital_los_days IS NULL THEN 1 END) AS missing_los,
  CASE
    WHEN COUNT(CASE WHEN gender IS NULL THEN 1 END) = 0
     AND COUNT(CASE WHEN age_at_admission IS NULL THEN 1 END) = 0
    THEN '✅ PASS'
    ELSE '❌ FAIL'
  END AS status
FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`;


-- ============================================================================
-- QC 2: Distribution & Range Validation
--   Goal: Ensure age & LOS are within clinically plausible ranges.
-- ============================================================================

SELECT
  'QC2_Distribution' AS check_name,
  COUNT(*) AS total_rows,
  ROUND(AVG(age_at_admission), 1) AS avg_age,
  ROUND(MIN(age_at_admission), 1) AS min_age,
  ROUND(MAX(age_at_admission), 1) AS max_age,
  ROUND(AVG(hospital_los_days), 1) AS avg_los,
  ROUND(MIN(hospital_los_days), 1) AS min_los,
  ROUND(MAX(hospital_los_days), 1) AS max_los,
  CASE
    WHEN MIN(age_at_admission) >= 18 AND MAX(age_at_admission) <= 120
    THEN '✅ PASS'
    ELSE '❌ FAIL'
  END AS age_range_status
FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`;


-- ============================================================================
-- QC 3: Demographic Summary by Gender
--   Goal: Check gender distribution and readmission rates.
-- ============================================================================

SELECT
  'Demographics Summary' AS section,
  gender,
  COUNT(*) AS count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
  ROUND(AVG(age_at_admission), 1) AS avg_age,
  SUM(CASE WHEN readmit_30d_flag = 1 THEN 1 ELSE 0 END) AS readmitted_count,
  ROUND(
    100.0 * SUM(CASE WHEN readmit_30d_flag = 1 THEN 1 ELSE 0 END) / COUNT(*),
    1
  ) AS readmit_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
GROUP BY gender
ORDER BY count DESC;


-- ============================================================================
-- QC 4: Overall Readmission Analysis
--   Goal: Confirm overall readmission rate using this feature layer.
-- ============================================================================

SELECT
  'Readmission_Analysis' AS analysis_name,
  COUNT(*) AS total_patients,
  SUM(readmit_30d_flag) AS readmitted_count,
  COUNT(*) - SUM(readmit_30d_flag) AS not_readmitted_count,
  ROUND(100.0 * SUM(readmit_30d_flag) / COUNT(*), 2) AS readmission_rate_pct,
  ROUND(
    100.0 * (COUNT(*) - SUM(readmit_30d_flag)) / COUNT(*),
    2
  ) AS no_readmission_rate_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`;






/*==============================================================================
PROJECT : ICU 30-Day Readmission – Anthropometry Feature Layer
TABLE   : feature_anthropometry
AUTHOR  : Jyoti Prakash Das
VERSION : 1.0 (multi-source, unit-normalized, production-ready)
LAST UPDATED: 2025-12-10

PURPOSE
-------
We built this table to provide **clean, clinically plausible height, weight,
and BMI** for every index ICU stay in our readmission cohort.

Why this layer was important:
  - Real-world hospital data stores anthropometry in many places (derived tables,
    ICU flowsheets, OMR text, input/procedure weights).
  - We needed robust logic to:
      • merge multiple sources with a clear priority,
      • normalize units (cm/kg vs inches/pounds/meters),
      • filter out obviously incorrect values,
      • compute BMI safely for adults only.

This table gives a **single, trusted version** of height/weight/BMI per index
stay, plus availability flags to track missingness in our ML pipeline.

DRIVER COHORT
-------------
nomadic-freedom-436306-g4.readmission30.feature_demographics_extended
  - One row per index ICU stay
  - Keys used: subject_id, hadm_id, index_stay_id, index_icu_intime

DATA SOURCES (Priority Logic)
-----------------------------
1) Derived first-day:
   - physionet-data.mimiciv_3_1_derived.first_day_height
   - physionet-data.mimiciv_3_1_derived.first_day_weight
   → High-quality, already cleaned; preferred when present.

2) ICU chartevents (flowsheets):
   - physionet-data.mimiciv_3_1_icu.chartevents
   - Selected anthropometry itemids:
       • 226730 – height (cm)
       • 226707 – height (raw, often inches or meters)
       • 226512 – weight (kg)
       • 226531 – weight (lb)
       • 224639 – daily weight
       • 226846 – feeding weight
   → We took the most recent value **before** index_icu_intime.

3) Input / Procedure weights:
   - physionet-data.mimiciv_3_1_icu.inputevents (patientweight)
   - physionet-data.mimiciv_3_1_icu.procedureevents (patientweight)
   → Used as additional fallbacks; often in kg.

4) Hospital OMR (textual measurements):
   - physionet-data.mimiciv_3_1_hosp.omr
   → We parsed height/weight/BMI from text using regex and keyword-based
     unit detection (cm/inches/kg/lb).

UNIT NORMALIZATION & ADULT FILTERS
----------------------------------
We normalized everything to:
  - Height → centimeters (cm)
      • inches → cm (× 2.54)
      • meters → cm (× 100)
  - Weight → kilograms (kg)
      • pounds → kg (× 0.45359237)

We then applied **conservative adult thresholds**:
  - Height: keep only if 120 <= height_cm <= 220
  - Weight: keep only if 30 <= weight_kg <= 300
  - BMI:    keep only if 12 <= bmi <= 60

This avoided extreme outliers and pediatric values.

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Multi-source fusion with clear precedence
   - We prioritized derived first-day tables, then ICU flowsheets, then
     input/procedure weights, and finally OMR text.
   - This ensured that the “best” data were used whenever available.

2) Time-awareness for chartevents
   - We only used chartevents **before** index_icu_intime to reflect
     baseline anthropometry and avoid later noise.

3) Robust unit handling
   - We cross-checked units using:
       • known itemids (already in cm/kg vs lb/in),
       • detected_unit from OMR strings,
       • value ranges to infer meters vs cm vs inches.

4) Adult-only plausibility checks
   - We explicitly filtered to adult ranges, consistent with our cohort
     (age ≥ 18), so extreme or pediatric records did not pollute features.

5) BMI strategy
   - We used OMR BMI when present and plausible (12–60).
   - Otherwise, we computed BMI from normalized height_cm & weight_kg and
     re-validated the computed BMI against the same 12–60 range.

6) Availability flags
   - We added height/weight/BMI available flags to make it easy to:
       • track missingness,
       • condition models on data availability,
       • debug pipeline coverage.

──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

Core Keys
• subject_id
    → Unique patient ID. Used to join across all feature layers.
• hadm_id
    → Hospital admission ID. Anchors anthropometry to the admission.
• index_stay_id
    → Index ICU stay ID. Aligns anthropometry features with ICU-level data.

Anthropometry Features
• height_cm
    → Final adult height in centimeters (120–220 cm only).
      Why important: Underweight, obesity, and fluid status are all interpreted
      relative to height (e.g., BMI, predicted body weight, dosing).

• weight_kg
    → Final adult weight in kilograms (30–300 kg only).
      Why important: Higher baseline weight and extremes of body mass are
      associated with different risk profiles, medication dosing, and
      discharge planning – all of which link to readmission risk.

• bmi
    → Body Mass Index (kg/m²), either from trusted OMR BMI or computed from
      (height_cm, weight_kg), only kept if 12–60.
      Why important:
        - Low BMI may reflect frailty/malnutrition → higher readmission risk.
        - High BMI may reflect obesity with comorbidities (DM, OSA, HF, etc.).
        - BMI is a strong general risk indicator for outcomes.

Availability Flags
• height_available_flag
    → 1 if height_cm is non-null, else 0.
      Helps us understand height coverage and enables model features like
      “missingness indicators”.

• weight_available_flag
    → 1 if weight_kg is non-null, else 0.
      Useful for sensitivity analysis and data-quality monitoring.

• bmi_available_flag
    → 1 if bmi is non-null, else 0.
      Important because BMI is a derived feature; we might treat BMI-missing
      patients differently in modeling.

Metadata
• feature_extraction_timestamp
    → When this feature table was created. Critical for lineage and auditing.

Usage for Readmission Modeling
------------------------------
- Direct numeric predictors:
    • height_cm, weight_kg, bmi as continuous features.
- Categorical / binned usage in models or dashboards:
    • BMI buckets (underweight/normal/overweight/obese) derived downstream.
- As proxies for:
    • frailty, nutritional status, chronic disease burden,
      which are all linked to 30-day ICU readmission risk.

==============================================================================
CREATE TABLE: feature_anthropometry
==============================================================================*/

CREATE OR REPLACE TABLE
  `nomadic-freedom-436306-g4.readmission30.feature_anthropometry` AS

WITH
-- 0) COHORT: index ICU stays (canonical driver)
cohort AS (
  SELECT subject_id, hadm_id, index_stay_id, index_icu_intime
  FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
),

-- 1) Derived first-day (aggregate by subject for robustness)
derived_height AS (
  SELECT subject_id, ANY_VALUE(height) AS derived_height_cm
  FROM `physionet-data.mimiciv_3_1_derived.first_day_height`
  GROUP BY subject_id
),
derived_weight AS (
  SELECT subject_id, ANY_VALUE(weight) AS derived_weight_kg
  FROM `physionet-data.mimiciv_3_1_derived.first_day_weight`
  GROUP BY subject_id
),

-- 2) Recent ICU chartevents (most recent <= index ICU intime for candidate itemids)
chartevents_recent AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    ch.itemid,
    ch.valuenum,
    ch.charttime,
    ROW_NUMBER() OVER (
      PARTITION BY c.subject_id, ch.itemid
      ORDER BY ch.charttime DESC
    ) AS rn
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ch
    ON c.subject_id = ch.subject_id
   AND ch.charttime <= c.index_icu_intime
   AND ch.valuenum IS NOT NULL
   -- anthropometry itemids
   AND ch.itemid IN (224639, 226512, 226531, 226846, 226730, 226707)
),

chartevents_pivot AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    MAX(CASE WHEN itemid = 226730 AND rn = 1 THEN valuenum END) AS height_cm_ce,
    MAX(CASE WHEN itemid = 226707 AND rn = 1 THEN valuenum END) AS height_raw_ce,
    MAX(CASE WHEN itemid = 226512 AND rn = 1 THEN valuenum END) AS weight_kg_ce,
    MAX(CASE WHEN itemid = 226531 AND rn = 1 THEN valuenum END) AS weight_lb_ce,
    MAX(CASE WHEN itemid = 224639 AND rn = 1 THEN valuenum END) AS daily_weight_ce,
    MAX(CASE WHEN itemid = 226846 AND rn = 1 THEN valuenum END) AS feeding_weight_ce
  FROM chartevents_recent
  GROUP BY subject_id, hadm_id, index_stay_id
),

-- 3) Input / Procedure patientweight (fallbacks, stay-level)
input_patientweight AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(stay_id AS INT64) AS index_stay_id,
    ANY_VALUE(patientweight) AS input_patientweight_kg
  FROM `physionet-data.mimiciv_3_1_icu.inputevents`
  WHERE patientweight IS NOT NULL
  GROUP BY subject_id, hadm_id, stay_id
),
proc_patientweight AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(stay_id AS INT64) AS index_stay_id,
    ANY_VALUE(patientweight) AS proc_patientweight_kg
  FROM `physionet-data.mimiciv_3_1_icu.procedureevents`
  WHERE patientweight IS NOT NULL
  GROUP BY subject_id, hadm_id, stay_id
),

-- 4) OMR textual parse (height / weight / BMI) – subject-level only
omr_parsed AS (
  SELECT
    o.subject_id,
    LOWER(o.result_name) AS result_name,
    o.result_value,
    SAFE_CAST(REGEXP_REPLACE(o.result_value, r'[^0-9.\-]', '') AS FLOAT64) AS result_num,
    CASE
      WHEN LOWER(o.result_value) LIKE '%kg%' THEN 'kg'
      WHEN LOWER(o.result_value) LIKE '%lb%' OR LOWER(o.result_value) LIKE '%lbs%' OR LOWER(o.result_value) LIKE '%pound%' THEN 'lb'
      WHEN LOWER(o.result_value) LIKE '%cm%' THEN 'cm'
      WHEN LOWER(o.result_value) LIKE '%in%' OR o.result_value LIKE '%"' OR LOWER(o.result_value) LIKE '%inch%' OR LOWER(o.result_value) LIKE '%ft%' THEN 'in'
      ELSE NULL
    END AS detected_unit
  FROM `physionet-data.mimiciv_3_1_hosp.omr` o
  WHERE LOWER(result_name) LIKE '%weight%'
     OR LOWER(result_name) LIKE '%height%'
     OR LOWER(result_name) LIKE '%bmi%'
),

omr_agg AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    MAX(CASE WHEN result_name LIKE '%height%' THEN result_num END) AS height_num,
    MAX(CASE WHEN result_name LIKE '%height%' THEN detected_unit END) AS height_unit,
    MAX(CASE WHEN result_name LIKE '%weight%' THEN result_num END) AS weight_num,
    MAX(CASE WHEN result_name LIKE '%weight%' THEN detected_unit END) AS weight_unit,
    MAX(CASE WHEN result_name LIKE '%bmi%' THEN result_num END)     AS omr_bmi
  FROM cohort c
  LEFT JOIN omr_parsed o
    ON c.subject_id = o.subject_id
  GROUP BY c.subject_id, c.hadm_id, c.index_stay_id
),

-- 5) Assemble candidate sources into one row per index_stay
assembled AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    dh.derived_height_cm,
    dw.derived_weight_kg,
    ce.height_cm_ce,
    ce.height_raw_ce,
    ce.weight_kg_ce,
    ce.weight_lb_ce,
    ce.daily_weight_ce,
    ce.feeding_weight_ce,
    ip.input_patientweight_kg,
    pp.proc_patientweight_kg,
    o.height_num,
    o.height_unit,
    o.weight_num,
    o.weight_unit,
    o.omr_bmi
  FROM cohort c
  LEFT JOIN derived_height dh
    ON c.subject_id = dh.subject_id
  LEFT JOIN derived_weight dw
    ON c.subject_id = dw.subject_id
  LEFT JOIN chartevents_pivot ce
    ON c.subject_id    = ce.subject_id
   AND c.hadm_id       = ce.hadm_id
   AND c.index_stay_id = ce.index_stay_id
  LEFT JOIN input_patientweight ip
    ON c.subject_id    = ip.subject_id
   AND c.hadm_id       = ip.hadm_id
   AND c.index_stay_id = ip.index_stay_id
  LEFT JOIN proc_patientweight pp
    ON c.subject_id    = pp.subject_id
   AND c.hadm_id       = pp.hadm_id
   AND c.index_stay_id = pp.index_stay_id
  LEFT JOIN omr_agg o
    ON c.subject_id    = o.subject_id
   AND c.hadm_id       = o.hadm_id
   AND c.index_stay_id = o.index_stay_id
),

-- 6) Normalize units and apply intermediate plausibility checks
normalized AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,

    -- Height → cm (intermediate)
    CASE
      WHEN derived_height_cm IS NOT NULL AND derived_height_cm BETWEEN 50 AND 250 THEN derived_height_cm
      WHEN height_cm_ce     IS NOT NULL AND height_cm_ce     BETWEEN 50 AND 250 THEN height_cm_ce
      WHEN height_raw_ce    IS NOT NULL AND height_raw_ce    BETWEEN 50 AND 84  THEN ROUND(height_raw_ce * 2.54, 2)    -- inches → cm
      WHEN height_raw_ce    IS NOT NULL AND height_raw_ce    BETWEEN 1.0 AND 2.5 THEN ROUND(height_raw_ce * 100.0, 2)  -- meters → cm
      WHEN height_num IS NOT NULL
           AND LOWER(COALESCE(height_unit, '')) = 'cm'
           AND height_num BETWEEN 50 AND 250 THEN height_num
      WHEN height_num IS NOT NULL
           AND LOWER(COALESCE(height_unit, '')) = 'in'
           AND height_num BETWEEN 20 AND 84 THEN ROUND(height_num * 2.54, 2)    -- inches → cm
      WHEN height_num IS NOT NULL
           AND height_num BETWEEN 1.0 AND 2.5 THEN ROUND(height_num * 100.0, 2) -- meters → cm (no explicit unit)
      ELSE NULL
    END AS raw_height_cm,

    -- Weight → kg (intermediate)
    CASE
      WHEN derived_weight_kg      IS NOT NULL AND derived_weight_kg      BETWEEN 2 AND 500   THEN ROUND(derived_weight_kg, 2)
      WHEN weight_kg_ce           IS NOT NULL AND weight_kg_ce           BETWEEN 2 AND 500   THEN ROUND(weight_kg_ce, 2)
      WHEN weight_lb_ce           IS NOT NULL AND weight_lb_ce           BETWEEN 20 AND 1000 THEN ROUND(weight_lb_ce * 0.45359237, 2)
      WHEN daily_weight_ce        IS NOT NULL AND daily_weight_ce        BETWEEN 2 AND 500   THEN ROUND(daily_weight_ce, 2)
      WHEN feeding_weight_ce      IS NOT NULL AND feeding_weight_ce      BETWEEN 2 AND 500   THEN ROUND(feeding_weight_ce, 2)
      WHEN input_patientweight_kg IS NOT NULL AND input_patientweight_kg BETWEEN 2 AND 500   THEN ROUND(input_patientweight_kg, 2)
      WHEN proc_patientweight_kg  IS NOT NULL AND proc_patientweight_kg  BETWEEN 2 AND 500   THEN ROUND(proc_patientweight_kg, 2)
      WHEN weight_num IS NOT NULL
           AND LOWER(COALESCE(weight_unit, '')) = 'kg'
           AND weight_num BETWEEN 2 AND 500 THEN ROUND(weight_num, 2)
      WHEN weight_num IS NOT NULL
           AND LOWER(COALESCE(weight_unit, '')) = 'lb'
           AND weight_num BETWEEN 20 AND 1000 THEN ROUND(weight_num * 0.45359237, 2)
      WHEN weight_num IS NOT NULL
           AND weight_num BETWEEN 20 AND 500 THEN ROUND(weight_num, 2)
      ELSE NULL
    END AS raw_weight_kg,

    omr_bmi
  FROM assembled
),

-- 7) FINAL: apply production thresholds and compute BMI (adult-only)
final_clean AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,

    -- Final height: adult 120–220 cm
    CASE
      WHEN raw_height_cm BETWEEN 120 AND 220 THEN ROUND(raw_height_cm, 2)
      ELSE NULL
    END AS height_cm,

    -- Final weight: adult 30–300 kg
    CASE
      WHEN raw_weight_kg BETWEEN 30 AND 300 THEN ROUND(raw_weight_kg, 2)
      ELSE NULL
    END AS weight_kg,

    -- BMI: prefer OMR; otherwise compute from height & weight
    CASE
      WHEN omr_bmi IS NOT NULL AND omr_bmi BETWEEN 12 AND 60 THEN ROUND(omr_bmi, 2)
      WHEN raw_height_cm BETWEEN 120 AND 220
       AND raw_weight_kg BETWEEN 30 AND 300 THEN
        CASE
          WHEN ROUND(raw_weight_kg / POWER(raw_height_cm / 100.0, 2), 2) BETWEEN 12 AND 60
          THEN ROUND(raw_weight_kg / POWER(raw_height_cm / 100.0, 2), 2)
          ELSE NULL
        END
      ELSE NULL
    END AS bmi
  FROM normalized
)

SELECT
  subject_id,
  hadm_id,
  index_stay_id,
  height_cm,
  weight_kg,
  bmi,
  -- Availability flags
  CASE WHEN height_cm IS NOT NULL THEN 1 ELSE 0 END AS height_available_flag,
  CASE WHEN weight_kg IS NOT NULL THEN 1 ELSE 0 END AS weight_available_flag,
  CASE WHEN bmi       IS NOT NULL THEN 1 ELSE 0 END AS bmi_available_flag,
  CURRENT_TIMESTAMP() AS feature_extraction_timestamp
FROM final_clean
ORDER BY subject_id;





/*==============================================================================
QC BLOCK – feature_anthropometry
We added these QC queries to verify completeness, plausibility, and distribution
of height, weight, and BMI for the cohort.
==============================================================================*/

-- ============================================================================
-- QC 1: Row Count & Coverage (height / weight / BMI availability)
-- ============================================================================

SELECT
  'QC1_Anthropometry_Completeness' AS check_name,
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN height_cm IS NOT NULL THEN 1 END) AS has_height,
  COUNT(CASE WHEN weight_kg IS NOT NULL THEN 1 END) AS has_weight,
  COUNT(CASE WHEN bmi IS NOT NULL THEN 1 END)       AS has_bmi,
  ROUND(100.0 * COUNT(CASE WHEN bmi IS NOT NULL THEN 1 END) / COUNT(*), 1) AS bmi_coverage_pct,
  CASE
    WHEN ROUND(100.0 * COUNT(CASE WHEN bmi IS NOT NULL THEN 1 END) / COUNT(*), 1) > 85
    THEN '✅ PASS'
    ELSE '⚠️  WARNING – Check source coverage'
  END AS status
FROM `nomadic-freedom-436306-g4.readmission30.feature_anthropometry`;


-- ============================================================================
-- QC 2: Clinical Range Validation
--   Goal: ensure final cleaned features stay within adult plausible ranges.
-- ============================================================================

SELECT
  'QC2_Clinical_Ranges' AS check_name,
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN height_cm IS NOT NULL AND (height_cm < 120 OR height_cm > 220) THEN 1 END) AS out_of_range_height,
  COUNT(CASE WHEN weight_kg IS NOT NULL AND (weight_kg < 30 OR weight_kg > 300) THEN 1 END)   AS out_of_range_weight,
  COUNT(CASE WHEN bmi       IS NOT NULL AND (bmi < 12 OR bmi > 60) THEN 1 END)               AS out_of_range_bmi,
  CASE
    WHEN COUNT(CASE WHEN height_cm IS NOT NULL AND (height_cm < 120 OR height_cm > 220) THEN 1 END) = 0
     AND COUNT(CASE WHEN weight_kg IS NOT NULL AND (weight_kg < 30 OR weight_kg > 300) THEN 1 END)   = 0
     AND COUNT(CASE WHEN bmi       IS NOT NULL AND (bmi < 12 OR bmi > 60) THEN 1 END)               = 0
    THEN '✅ PASS'
    ELSE '⚠️  Check unexpected outliers'
  END AS status
FROM `nomadic-freedom-436306-g4.readmission30.feature_anthropometry`;


-- ============================================================================
-- QC 3: Distribution Statistics
--   Goal: get a quick sense of typical height/weight/BMI in this cohort.
-- ============================================================================

SELECT
  'QC3_Distribution' AS check_name,
  ROUND(AVG(height_cm), 1) AS avg_height_cm,
  ROUND(MIN(height_cm), 1) AS min_height_cm,
  ROUND(MAX(height_cm), 1) AS max_height_cm,
  ROUND(STDDEV(height_cm), 1) AS stddev_height_cm,
  ROUND(AVG(weight_kg), 1) AS avg_weight_kg,
  ROUND(MIN(weight_kg), 1) AS min_weight_kg,
  ROUND(MAX(weight_kg), 1) AS max_weight_kg,
  ROUND(STDDEV(weight_kg), 1) AS stddev_weight_kg,
  ROUND(AVG(bmi), 2) AS avg_bmi,
  ROUND(MIN(bmi), 2) AS min_bmi,
  ROUND(MAX(bmi), 2) AS max_bmi
FROM `nomadic-freedom-436306-g4.readmission30.feature_anthropometry`
WHERE bmi IS NOT NULL;



-------- HAVE TO CREATE A MAPPING CSV FILES or direct codes from verifed sources FOR VITALS AND LAB TEST RESULTS I THINK also use preddeined tbales from derived OTHERWISe many errors

-- ============================================================================
-- DIAGNOSTIC: Get ACTUAL column names from MIMIC-IV derived tables
-- ============================================================================

-- First_day_vitalsign schema
SELECT
  column_name,
  data_type,
  is_nullable
FROM `physionet-data.mimiciv_3_1_derived`.INFORMATION_SCHEMA.COLUMNS
WHERE table_name = 'first_day_vitalsign'
ORDER BY ordinal_position;

-- First_day_lab schema
SELECT
  column_name,
  data_type,
  is_nullable
FROM `physionet-data.mimiciv_3_1_derived`.INFORMATION_SCHEMA.COLUMNS
WHERE table_name = 'first_day_lab'
ORDER BY ordinal_position;


/* ==============================================================================
PROJECT : ICU 30-Day Readmission – Vitals (First 24h) Feature Layer
TABLE   : feature_vitals_first_24h
AUTHOR  : Jyoti Prakash Das
VERSION : 1.0 (derived + chartevents fallback, production-ready)
LAST UPDATED: 2025-12-10

PURPOSE
-------
We built this table to summarize first-24-hour vitals for each index ICU stay
in our 30-day readmission cohort.

This layer provides:
  - Clean, stay-level heart rate, blood pressure, temperature, respiratory rate,
    oxygen saturation, and glucose summaries.
  - Ventilation-related features (FiO2, PEEP, tidal volume, plateau, mech vent flag).
  - Shock marker features (MAP < 65).
These are strong early clinical severity signals for 30-day ICU readmission risk.

DRIVER COHORT
-------------
nomadic-freedom-436306-g4.readmission30.feature_demographics_extended
  - One row per index ICU stay.
  - Keys used: subject_id, hadm_id, index_stay_id, index_icu_intime.

DATA SOURCES
------------
1) physionet-data.mimiciv_3_1_derived.first_day_vitalsign
   - Canonical first-24h vitals aggregated by ICU stay (stay_id).
   - We used mean, min, max for heart rate, temperature, SBP, DBP, MAP, RR, SpO2, glucose.

2) physionet-data.mimiciv_3_1_icu.chartevents
   - Used as fallback for:
       - FiO2 percent (itemid 223835)
       - Mechanical ventilation flag (itemid 223849)
       - PEEP (224700)
       - PIP (224695)
       - Tidal volume (224685)
       - Plateau pressure (224696)
   - Restricted to 0–24 hours after index_icu_intime when stay_id is missing.

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Derived vitals as primary source
   - We relied on first_day_vitalsign as the main source since it is pre-validated
     and stay-level. This reduced complexity and avoided repeated low-level chart parsing.

2) Chartevents only for specific ventilation-related fields
   - We used chartevents only for variables not present in first_day_vitalsign:
     FiO2, ventilation flags, PEEP, PIP, tidal volume, plateau pressure.
   - We limited to numeric valuenum and itemids of interest.

3) Time-aware chartevents window
   - When stay_id was available on chartevents, we matched by stay_id.
   - When stay_id was missing, we joined by (subject_id, hadm_id) and restricted
     charttime to [index_icu_intime, index_icu_intime + 24 hours).

4) Plausibility filters on chartevents values
   - We applied conservative filters to drop impossible or clearly erroneous values:
       - FiO2 percent kept only in [21, 100].
       - PEEP, PIP, plateau pressure kept in [0, 80].
       - Tidal volume kept in [20, 5000] mL.

5) Shock and glycemic flags
   - We derived:
       - shock_flag_map_lt_65: mean MAP < 65 mmHg in first 24h.
       - hypoglycemia_flag: glucose_min < 70.
       - hyperglycemia_flag: glucose_max > 180.

6) Availability flags
   - We created hr_available_flag, temp_available_flag, bp_available_flag,
     spo2_available_flag, fio2_available_flag for downstream missingness-aware models.

OUTPUT COLUMNS (short definitions and why they matter)
------------------------------------------------------

Key identifiers
- subject_id
  Unique patient identifier; used to join with other feature layers.

- hadm_id
  Hospital admission identifier; keeps vitals aligned with admission-level data.

- index_stay_id
  ICU stay identifier for the index stay; primary join key to derived tables.

Heart rate features
- hr_first_24h_mean
  Mean heart rate over first 24h. Higher values often reflect shock, sepsis,
  or pain; strong risk signal.

- hr_first_24h_min
  Lowest heart rate; can reflect bradycardia due to drugs, conduction disease,
  or measurement artifacts.

- hr_first_24h_max
  Peak heart rate; indicates stress response, sepsis, hemorrhage, arrhythmia.

- hr_first_24h_range
  Difference between max and min heart rate; a crude variability proxy and
  marker of instability.

Temperature features
- temp_c_first_24h_mean
- temp_c_first_24h_min
- temp_c_first_24h_max
  First-24h temperature in degrees Celsius; extreme fever or hypothermia is
  associated with infection, sepsis, or poor prognosis.

Blood pressure and shock marker
- sbp_first_24h_mean
  Mean systolic BP; hypotension patterns relate to shock and organ hypoperfusion.

- sbp_first_24h_min
  Nadir systolic BP; captures transient hypotensive episodes.

- dbp_first_24h_mean
  Mean diastolic BP; helps characterize vascular tone and perfusion.

- mbp_first_24h_mean
  Mean arterial pressure (MAP) over first 24h.

- shock_flag_map_lt_65
  Binary flag (1/0), 1 if mbp_first_24h_mean < 65. Classic shock/perfusion
  threshold; patients with low MAP are at higher risk of complications and
  readmission.

Respiratory rate features
- rr_first_24h_mean
- rr_first_24h_max
- rr_first_24h_min
  Breathing pattern in first 24h; tachypnea is often an early sign of
  decompensation.

Oxygenation features
- spo2_first_24h_mean
- spo2_first_24h_min
- spo2_first_24h_max
  First-24h oxygen saturation; low minima are markers of respiratory failure.

- fio2_percent_first_24h_mean
  Mean FiO2 in percent from chartevents. Higher FiO2 requirements mean the
  patient needed more oxygen support.

- sf_ratio_approx
  Approximate SpO2/FiO2 ratio. Lower values indicate worse oxygenation; a
  non-invasive surrogate for PaO2/FiO2.

Glucose features
- glucose_first_24h_mean
- glucose_first_24h_min
- glucose_first_24h_max
  First-24h glucose profile; both hypo- and hyperglycemia are linked to worse
  outcomes and readmission risk.

- hypoglycemia_flag
  1 if glucose_min < 70 mg/dL; identifies patients with significant low sugars.

- hyperglycemia_flag
  1 if glucose_max > 180 mg/dL; captures stress hyperglycemia and poor control.

Ventilation and lung mechanics (from chartevents)
- peep_first_24h_mean
  Mean PEEP in cmH2O. Higher PEEP reflects severe lung pathology or ARDS.

- pip_first_24h_mean
  Mean peak inspiratory pressure; relates to airway resistance and lung mechanics.

- tv_first_24h_mean
  Mean tidal volume in mL; helps understand ventilation strategy (protective vs not).

- plateau_first24h_mean
  Mean plateau pressure; used to assess lung compliance and risk of barotrauma.

- mechvent_first24h_flag
  1 if mechanical ventilation chart flag present and non-zero; indicates patients
  requiring invasive ventilatory support.

Availability flags
- hr_available_flag
- temp_available_flag
- bp_available_flag
- spo2_available_flag
- fio2_available_flag
  1/0 flags indicating whether each major vital sign block is present. Helpful for
  understanding coverage, building missingness-aware models, and debugging pipelines.

Metadata
- feature_extraction_timestamp
  Timestamp when this vitals feature table was created (for lineage and auditing).

==============================================================================*/


-- ==============================================================================
-- CREATE TABLE: feature_vitals_first_24h
-- ==============================================================================

CREATE OR REPLACE TABLE `nomadic-freedom-436306-g4.readmission30.feature_vitals_first_24h` AS

WITH
-- 1) Cohort: index ICU stays (driver)
cohort AS (
  SELECT subject_id, hadm_id, index_stay_id, index_icu_intime
  FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
),

-- 2) Derived vitals (stay-level aggregates)
derived_v AS (
  SELECT
    stay_id,
    subject_id,
    SAFE_CAST(heart_rate_mean  AS FLOAT64) AS heart_rate_mean,
    SAFE_CAST(heart_rate_min   AS FLOAT64) AS heart_rate_min,
    SAFE_CAST(heart_rate_max   AS FLOAT64) AS heart_rate_max,

    SAFE_CAST(temperature_mean AS FLOAT64) AS temperature_mean,
    SAFE_CAST(temperature_min  AS FLOAT64) AS temperature_min,
    SAFE_CAST(temperature_max  AS FLOAT64) AS temperature_max,

    SAFE_CAST(sbp_mean         AS FLOAT64) AS sbp_mean,
    SAFE_CAST(sbp_min          AS FLOAT64) AS sbp_min,
    SAFE_CAST(sbp_max          AS FLOAT64) AS sbp_max,

    SAFE_CAST(dbp_mean         AS FLOAT64) AS dbp_mean,
    SAFE_CAST(dbp_min          AS FLOAT64) AS dbp_min,
    SAFE_CAST(dbp_max          AS FLOAT64) AS dbp_max,

    SAFE_CAST(mbp_mean         AS FLOAT64) AS mbp_mean,

    SAFE_CAST(resp_rate_mean   AS FLOAT64) AS resp_rate_mean,
    SAFE_CAST(resp_rate_min    AS FLOAT64) AS resp_rate_min,
    SAFE_CAST(resp_rate_max    AS FLOAT64) AS resp_rate_max,

    SAFE_CAST(spo2_mean        AS FLOAT64) AS spo2_mean,
    SAFE_CAST(spo2_min         AS FLOAT64) AS spo2_min,
    SAFE_CAST(spo2_max         AS FLOAT64) AS spo2_max,

    SAFE_CAST(glucose_mean     AS FLOAT64) AS glucose_mean,
    SAFE_CAST(glucose_min      AS FLOAT64) AS glucose_min,
    SAFE_CAST(glucose_max      AS FLOAT64) AS glucose_max

  FROM `physionet-data.mimiciv_3_1_derived.first_day_vitalsign`
),

-- 3) Chartevents fallback: ventilation-related fields in first 24h
chartevents_24h AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    SAFE_CAST(ch.stay_id AS INT64) AS ch_stay_id,
    SAFE_CAST(ch.itemid AS INT64) AS itemid,
    SAFE_CAST(ch.valuenum AS FLOAT64) AS valuenum,
    ch.charttime
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ch
    ON (
         -- primary: exact stay mapping when stay_id present
         (SAFE_CAST(ch.stay_id AS INT64) IS NOT NULL
          AND SAFE_CAST(ch.stay_id AS INT64) = c.index_stay_id)
       OR
         -- fallback: subject+hadm and restricted to 0–24h window after ICU admit
         (SAFE_CAST(ch.stay_id AS INT64) IS NULL
          AND ch.subject_id = c.subject_id
          AND ch.hadm_id = c.hadm_id
          AND ch.charttime >= c.index_icu_intime
          AND ch.charttime < TIMESTAMP_ADD(c.index_icu_intime, INTERVAL 24 HOUR))
       )
  WHERE ch.valuenum IS NOT NULL
    AND ch.itemid IN (
      223835,   -- FiO2 percent
      223849,   -- mech vent flag
      224700,   -- PEEP
      224695,   -- PIP
      224685,   -- tidal volume (mL)
      224696    -- plateau pressure
    )
),

-- 4) Plausibility filters for chartevents values
chartevents_24h_clean AS (
  SELECT subject_id, hadm_id, index_stay_id, itemid, valuenum
  FROM chartevents_24h ch
  WHERE NOT (
        (ch.itemid = 223835 AND (ch.valuenum < 21 OR ch.valuenum > 100))           -- FiO2 percent
     OR (ch.itemid IN (224700, 224695, 224696) AND (ch.valuenum < 0 OR ch.valuenum > 80)) -- PEEP/PIP/plateau
     OR (ch.itemid = 224685 AND (ch.valuenum < 20 OR ch.valuenum > 5000))          -- tidal volume mL
  )
),

-- 5) Aggregate chartevents by stay and item
chartevents_agg AS (
  SELECT
    index_stay_id,
    subject_id,
    hadm_id,
    itemid,
    AVG(valuenum) AS mean_val,
    MIN(valuenum) AS min_val,
    MAX(valuenum) AS max_val,
    STDDEV_POP(valuenum) AS std_val,
    COUNT(1) AS n_obs
  FROM chartevents_24h_clean
  GROUP BY index_stay_id, subject_id, hadm_id, itemid
),

-- 6) Pivot chartevents into wide format (one row per stay)
chartevents_pivot AS (
  SELECT
    index_stay_id,
    subject_id,
    hadm_id,
    MAX(CASE WHEN itemid = 223835 THEN mean_val END) AS fio2_mean_ce,
    MAX(CASE WHEN itemid = 223849 THEN COALESCE(max_val, min_val, mean_val) END) AS mechvent_flag_ce,
    MAX(CASE WHEN itemid = 224700 THEN mean_val END) AS peep_mean_ce,
    MAX(CASE WHEN itemid = 224695 THEN mean_val END) AS pip_mean_ce,
    MAX(CASE WHEN itemid = 224685 THEN mean_val END) AS tv_mean_ce,
    MAX(CASE WHEN itemid = 224696 THEN mean_val END) AS plateau_mean_ce
  FROM chartevents_agg
  GROUP BY index_stay_id, subject_id, hadm_id
),

-- 7) Combine cohort, derived vitals, and chartevents pivot
combined AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,

    -- derived vitals (preferred)
    d.heart_rate_mean,
    d.heart_rate_min,
    d.heart_rate_max,

    d.temperature_mean,
    d.temperature_min,
    d.temperature_max,

    d.sbp_mean,
    d.sbp_min,
    d.sbp_max,

    d.dbp_mean,
    d.dbp_min,
    d.dbp_max,

    d.mbp_mean,

    d.resp_rate_mean,
    d.resp_rate_min,
    d.resp_rate_max,

    d.spo2_mean,
    d.spo2_min,
    d.spo2_max,

    d.glucose_mean,
    d.glucose_min,
    d.glucose_max,

    -- chartevents ventilation-related fields
    cp.fio2_mean_ce,
    cp.mechvent_flag_ce,
    cp.peep_mean_ce,
    cp.pip_mean_ce,
    cp.tv_mean_ce,
    cp.plateau_mean_ce

  FROM cohort c
  LEFT JOIN derived_v d
    ON c.index_stay_id = d.stay_id
  LEFT JOIN chartevents_pivot cp
    ON cp.index_stay_id = c.index_stay_id
)

-- 8) Final feature selection and derivations
SELECT
  subject_id,
  hadm_id,
  index_stay_id,

  -- Heart rate
  ROUND(heart_rate_mean, 2) AS hr_first_24h_mean,
  ROUND(heart_rate_min, 2)  AS hr_first_24h_min,
  ROUND(heart_rate_max, 2)  AS hr_first_24h_max,
  CASE
    WHEN heart_rate_max IS NOT NULL AND heart_rate_min IS NOT NULL
    THEN ROUND(heart_rate_max - heart_rate_min, 2)
    ELSE NULL
  END AS hr_first_24h_range,

  -- Temperature (Celsius)
  ROUND(temperature_mean, 2) AS temp_c_first_24h_mean,
  ROUND(temperature_min, 2)  AS temp_c_first_24h_min,
  ROUND(temperature_max, 2)  AS temp_c_first_24h_max,

  -- Blood pressure
  ROUND(sbp_mean, 2) AS sbp_first_24h_mean,
  ROUND(sbp_min, 2)  AS sbp_first_24h_min,
  ROUND(dbp_mean, 2) AS dbp_first_24h_mean,
  ROUND(mbp_mean, 2) AS mbp_first_24h_mean,

  -- Shock flag: MAP < 65
  CASE
    WHEN mbp_mean IS NOT NULL AND mbp_mean < 65 THEN 1
    ELSE 0
  END AS shock_flag_map_lt_65,

  -- Respiratory rate
  ROUND(resp_rate_mean, 2) AS rr_first_24h_mean,
  ROUND(resp_rate_max, 2)  AS rr_first_24h_max,
  ROUND(resp_rate_min, 2)  AS rr_first_24h_min,

  -- Oxygen saturation
  ROUND(spo2_mean, 2) AS spo2_first_24h_mean,
  ROUND(spo2_min, 2)  AS spo2_first_24h_min,
  ROUND(spo2_max, 2)  AS spo2_first_24h_max,

  -- FiO2 (percent) from chartevents
  ROUND(fio2_mean_ce, 4) AS fio2_first_24h_mean,

  -- SpO2 / FiO2 ratio approximation
  CASE
    WHEN fio2_mean_ce IS NOT NULL AND fio2_mean_ce > 0
    THEN ROUND((spo2_mean * 100.0) / fio2_mean_ce, 4)
    ELSE NULL
  END AS sf_ratio_approx,

  -- Glucose features
  ROUND(glucose_mean, 2) AS glucose_first_24h_mean,
  ROUND(glucose_min, 2)  AS glucose_first_24h_min,
  ROUND(glucose_max, 2)  AS glucose_first_24h_max,
  CASE WHEN glucose_min IS NOT NULL AND glucose_min < 70 THEN 1 ELSE 0 END AS hypoglycemia_flag,
  CASE WHEN glucose_max IS NOT NULL AND glucose_max > 180 THEN 1 ELSE 0 END AS hyperglycemia_flag,

  -- Ventilator settings
  ROUND(peep_mean_ce, 2)     AS peep_first_24h_mean,
  ROUND(pip_mean_ce, 2)      AS pip_first_24h_mean,
  ROUND(tv_mean_ce, 2)       AS tv_first_24h_mean,
  ROUND(plateau_mean_ce, 2)  AS plateau_first24h_mean,

  -- Mechanical ventilation flag (binary)
  CASE
    WHEN mechvent_flag_ce IS NOT NULL
         AND SAFE_CAST(mechvent_flag_ce AS FLOAT64) <> 0
    THEN 1
    ELSE 0
  END AS mechvent_first24h_flag,

  -- Availability flags
  CASE WHEN heart_rate_mean IS NOT NULL THEN 1 ELSE 0 END AS hr_available_flag,
  CASE WHEN temperature_mean IS NOT NULL THEN 1 ELSE 0 END AS temp_available_flag,
  CASE WHEN sbp_mean IS NOT NULL OR mbp_mean IS NOT NULL THEN 1 ELSE 0 END AS bp_available_flag,
  CASE WHEN spo2_mean IS NOT NULL THEN 1 ELSE 0 END AS spo2_available_flag,
  CASE WHEN fio2_mean_ce IS NOT NULL THEN 1 ELSE 0 END AS fio2_available_flag,

  CURRENT_TIMESTAMP() AS feature_extraction_timestamp

FROM combined
ORDER BY subject_id;

-- ==============================================================================
-- QC 1: Completeness and core vitals coverage
-- ==============================================================================

SELECT
  'QC1_Vitals_Completeness' AS check_name,
  COUNT(*) AS total_rows,

  COUNTIF(hr_first_24h_mean IS NOT NULL) AS has_hr,
  ROUND(100.0 * COUNTIF(hr_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_hr,

  COUNTIF(temp_c_first_24h_mean IS NOT NULL) AS has_temp,
  ROUND(100.0 * COUNTIF(temp_c_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_temp,

  COUNTIF(sbp_first_24h_mean IS NOT NULL OR mbp_first_24h_mean IS NOT NULL) AS has_bp,
  ROUND(100.0 * COUNTIF(sbp_first_24h_mean IS NOT NULL OR mbp_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_bp,

  COUNTIF(rr_first_24h_mean IS NOT NULL) AS has_rr,
  ROUND(100.0 * COUNTIF(rr_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_rr,

  COUNTIF(spo2_first_24h_mean IS NOT NULL) AS has_spo2,
  ROUND(100.0 * COUNTIF(spo2_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_spo2,

  COUNTIF(fio2_first_24h_mean IS NOT NULL) AS has_fio2,
  ROUND(100.0 * COUNTIF(fio2_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_fio2,

  COUNTIF(glucose_first_24h_mean IS NOT NULL) AS has_glucose,
  ROUND(100.0 * COUNTIF(glucose_first_24h_mean IS NOT NULL) / COUNT(*), 1) AS pct_has_glucose,

  COUNTIF(mechvent_first24h_flag = 1) AS n_mechvent_flag,
  ROUND(100.0 * COUNTIF(mechvent_first24h_flag = 1) / COUNT(*), 2) AS pct_mechvent,

  COUNTIF(
    hr_first_24h_mean IS NOT NULL
    AND temp_c_first_24h_mean IS NOT NULL
    AND (sbp_first_24h_mean IS NOT NULL OR mbp_first_24h_mean IS NOT NULL)
    AND rr_first_24h_mean IS NOT NULL
    AND spo2_first_24h_mean IS NOT NULL
  ) AS n_complete_core,
  ROUND(
    100.0 * COUNTIF(
      hr_first_24h_mean IS NOT NULL
      AND temp_c_first_24h_mean IS NOT NULL
      AND (sbp_first_24h_mean IS NOT NULL OR mbp_first_24h_mean IS NOT NULL)
      AND rr_first_24h_mean IS NOT NULL
      AND spo2_first_24h_mean IS NOT NULL
    ) / COUNT(*),
    1
  ) AS pct_complete_core
FROM `nomadic-freedom-436306-g4.readmission30.feature_vitals_first_24h`;


-- ==============================================================================
-- QC 2: Clinical range validation (simple counts)
-- ==============================================================================

SELECT
  'QC2_Clinical_Ranges' AS check_name,
  COUNT(*) AS total_rows,

  COUNTIF(hr_first_24h_mean < 30 OR hr_first_24h_mean > 250) AS abnormal_hr,
  COUNTIF(temp_c_first_24h_mean < 25 OR temp_c_first_24h_mean > 42) AS abnormal_temp,
  COUNTIF(sbp_first_24h_mean < 40 OR sbp_first_24h_mean > 300) AS abnormal_sbp,
  COUNTIF(mbp_first_24h_mean IS NOT NULL AND (mbp_first_24h_mean < 20 OR mbp_first_24h_mean > 200)) AS abnormal_mbp,
  COUNTIF(rr_first_24h_mean < 5 OR rr_first_24h_mean > 80) AS abnormal_rr,
  COUNTIF(spo2_first_24h_mean < 40 OR spo2_first_24h_mean > 100) AS abnormal_spo2,
  COUNTIF(fio2_first_24h_mean IS NOT NULL AND (fio2_first_24h_mean < 21 OR fio2_first_24h_mean > 100)) AS abnormal_fio2,
  COUNTIF(glucose_first_24h_min IS NOT NULL AND (glucose_first_24h_min < 1 OR glucose_first_24h_max > 2000)) AS abnormal_glucose,
  COUNTIF(peep_first_24h_mean IS NOT NULL AND (peep_first_24h_mean < 0 OR peep_first_24h_mean > 80)) AS abnormal_peep,
  COUNTIF(pip_first_24h_mean IS NOT NULL AND (pip_first_24h_mean < 0 OR pip_first_24h_mean > 80)) AS abnormal_pip,
  COUNTIF(tv_first_24h_mean IS NOT NULL AND (tv_first_24h_mean < 20 OR tv_first_24h_mean > 5000)) AS abnormal_tv,
  COUNTIF(mechvent_first24h_flag NOT IN (0, 1)) AS abnormal_mechvent_flag
FROM `nomadic-freedom-436306-g4.readmission30.feature_vitals_first_24h`;


-- ==============================================================================
-- QC 3: Internal consistency (min <= max)
-- ==============================================================================

SELECT
  'QC3_Min_Leq_Max' AS check_name,
  COUNT(*) AS total_rows,
  COUNTIF(hr_first_24h_min IS NOT NULL AND hr_first_24h_max IS NOT NULL AND hr_first_24h_min > hr_first_24h_max) AS hr_min_gt_max,
  COUNTIF(temp_c_first_24h_min IS NOT NULL AND temp_c_first_24h_max IS NOT NULL AND temp_c_first_24h_min > temp_c_first_24h_max) AS temp_min_gt_max,
  COUNTIF(rr_first_24h_min IS NOT NULL AND rr_first_24h_max IS NOT NULL AND rr_first_24h_min > rr_first_24h_max) AS rr_min_gt_max
FROM `nomadic-freedom-436306-g4.readmission30.feature_vitals_first_24h`;


-- ==============================================================================
-- QC 4: Distribution and shock / ventilation prevalence summary
-- ==============================================================================

SELECT
  'QC4_Distribution_Shock' AS check_name,
  ROUND(AVG(hr_first_24h_mean), 1) AS avg_hr,
  ROUND(MIN(hr_first_24h_mean), 1) AS min_hr,
  ROUND(MAX(hr_first_24h_mean), 1) AS max_hr,

  ROUND(AVG(temp_c_first_24h_mean), 2) AS avg_temp,
  ROUND(AVG(sbp_first_24h_mean), 1) AS avg_sbp,
  ROUND(AVG(mbp_first_24h_mean), 1) AS avg_mbp,
  ROUND(AVG(rr_first_24h_mean), 1) AS avg_rr,
  ROUND(AVG(spo2_first_24h_mean), 1) AS avg_spo2,

  COUNTIF(mbp_first_24h_mean IS NOT NULL AND mbp_first_24h_mean < 65) AS hypotensive_patients,
  ROUND(100.0 * COUNTIF(mbp_first_24h_mean IS NOT NULL AND mbp_first_24h_mean < 65) / COUNT(*), 2) AS hypotensive_pct,

  COUNTIF(mechvent_first24h_flag = 1) AS n_mechvent,
  ROUND(100.0 * COUNTIF(mechvent_first24h_flag = 1) / COUNT(*), 2) AS pct_mechvent
FROM `nomadic-freedom-436306-g4.readmission30.feature_vitals_first_24h`;



-- Well remeber for vital signs and lab results table we, first have to account for phsioloogical impposible outlier before taking it into python , i think i have done thsi for vital signs but need to do it with table 5 - we will do this in pyhotn ml pipeline 

/*==============================================================================
PROJECT : ICU 30-Day Readmission – First 24h Laboratory Features
TABLE   : feature_labs_first_24h
AUTHOR  : Jyoti Prakash Das
VERSION : 1.0 (derived + labevents fallback, production-grade)
LAST UPDATED: 2025-12-10

PURPOSE
-------
We built this table to capture **key lab markers from the first 24 hours of the
index ICU stay** for every patient in our readmission cohort.

Why this layer was important:
  - Early labs (first ICU day) reflect acute severity and organ dysfunction.
  - Many strong ICU risk signals live in labs: lactate, creatinine, sodium,
    platelets, INR, troponin, etc.
  - We needed:
      • robust use of physionet derived.first_day_lab when available,
      • a clean fallback using hosp.labevents in a 0–24h window,
      • simple, model-ready features (min/max + clinical flags).

DRIVER COHORT
-------------
nomadic-freedom-436306-g4.readmission30.feature_demographics_extended
  - One row per index ICU stay
  - Keys used here: subject_id, hadm_id, index_stay_id, index_icu_intime, readmit_30d_flag

DATA SOURCES (Priority Logic)
-----------------------------
1) physionet-data.mimiciv_3_1_derived.first_day_lab
   - Preferred source (already “first ICU day” summary per stay_id).
   - We parsed it via JSON and extracted numeric columns with SAFE_CAST.

2) physionet-data.mimiciv_3_1_hosp.labevents
   - Fallback for selected labs not present in derived table or to backfill.
   - Restricted to:
       • subject_id + hadm_id matching our cohort
       • charttime within [index_icu_intime, index_icu_intime + 24h)
       • key itemids (glucose, creatinine, BUN, lactate, troponin, Mg, Phos, CRP, pH, pCO2)
   - Aggregated per (subject_id, hadm_id, index_stay_id, itemid) into min/max/mean
     and first/last values over the 24h window.

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Derived-first, labevents-second
   - We always preferred derived.first_day_lab values when they existed.
   - We only fell back to labevents when the derived metric was NULL.
   - This gave us consistency with PhysioNet’s official pipelines while still
     recovering more coverage when needed.

2) 0–24h window anchored on index_icu_intime
   - Labevents fallback used an explicit time window:
       charttime >= index_icu_intime
       AND charttime < index_icu_intime + 24 hours
   - This kept the lab features aligned with the “first ICU day” clinical state.

3) Simple modeling schema
   - We focused on **min/max (and sometimes first/last)** values rather than
     complicated temporal trajectories.
   - We also created **binary clinical flags** (e.g., severe anemia, AKI,
     hyper/hypoglycemia, high lactate, low albumin) that are easy to interpret.

4) SAFE parsing & type handling
   - All JSON extraction used SAFE_CAST to avoid runtime errors.
   - Fallback lab values were restricted to valuenum (numeric) entries only.

5) Availability flags & fallback counters
   - We added *_available_flag columns to quantify coverage per lab family.
   - We tracked fallback observation counts for magnesium/phosphate/CRP for QC.

──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

Core Keys
• subject_id
    → Unique patient ID; used to join across feature layers.
• hadm_id
    → Hospital admission ID; ties features to the admission.
• index_stay_id
    → Index ICU stay ID; aligns labs with the correct ICU stay.
• readmit_30d_flag
    → Target label (0/1) for 30-day ICU readmission; included for convenience.

Hematology
• hemoglobin_first_24h_min / _max
    → Lowest and highest hemoglobin in first 24h.
      – Low values (e.g., <7) indicate severe anemia → we created severe_anemia_flag.
• hematocrit_first_24h_min / _max
    → Packed cell volume range; cross-checks anemia and hemodilution.
• platelets_first_24h_min / _max
    → Platelet counts; thrombocytopenia is linked to bleeding risk and severity.
• wbc_first_24h_max
    → Peak white count; elevated_wbc_flag marks strong inflammatory response/infection.

Glucose
• glucose_first_24h_min / _max
    → Hypoglycemia and hyperglycemia episodes in first ICU day.
• hypoglycemia_flag
    → 1 if glucose_min < 70 mg/dL; associated with harm and instability.
• hyperglycemia_flag
    → 1 if glucose_max > 180 mg/dL; reflects stress response / poor control.

Renal Function
• creatinine_first_24h_max
    → Peak creatinine (kidney injury). Used to define acute_kidney_injury_flag.
• bun_first_24h_max
    → Peak blood urea nitrogen; complements creatinine as renal / catabolic marker.
• acute_kidney_injury_flag
    → 1 if max creatinine > 2.0; crude AKI severity signal.

Electrolytes / Acid-Base
• sodium_first_24h_min / _max
    → Hyponatremia / hypernatremia → both associated with worse outcomes.
• potassium_first_24h_min / _max
    → Hypo/hyperkalemia (arrhythmia risk).
• chloride_first_24h_min
    → Hyperchloremia associated with worse kidney outcomes in some settings.
• bicarbonate_first_24h_min
    → Low bicarb reflects metabolic acidosis and shock risk.
• calcium_first_24h_min
    → Hypocalcemia may track severity, sepsis, transfusions.
• aniongap_first_24h_min
    → High anion gap suggests lactic acidosis / unmeasured anions.

Liver & Nutritional Status
• albumin_first_24h_min
    → Low albumin (low_albumin_flag) marks chronic illness, malnutrition, liver disease.
• low_albumin_flag
    → 1 if albumin < 2.0 g/dL; strong chronic disease / frailty signal.
• bilirubin_total_first_24h_max / bilirubin_direct_first_24h_max
    → Hepatic dysfunction and cholestasis markers.
• alt_first_24h_max / ast_first_24h_max
    → Hepatocellular injury markers (shock liver, hepatitis, etc.).

Coagulation & Thrombosis
• inr_first_24h_max, pt_first_24h_max, ptt_first_24h_max
    → Coagulation pathway status; derangements suggest liver failure, anticoagulation,
      DIC, or bleeding risk.
• d_dimer_first_24h_max
    → Elevated in clotting, DIC, sepsis; severity marker.
• fibrinogen_first_24h_max
    → Low in consumption coagulopathy; high in inflammation.

Perfusion / Shock
• lactate_first_24h_max
    → Peak lactate; elevated_lactate_flag (lactate > 4) is a classic shock severity marker.

Cardiac Markers
• troponin_first_24h_max
    → Myocardial injury (type 1/2 MI, demand ischemia); prognostic.
• ntprobnp_first_24h_max
    → Heart failure / volume load; important for readmission risk.

Differential White Counts
• abs_neutrophils_first_24h_max
• abs_lymphocytes_first_24h_max
• abs_monocytes_first_24h_max
    → Immune profile; neutrophilia vs lymphopenia patterns can be prognostic.

Electrolytes (Fallback Enrichment)
• magnesium_first_24h_max
• phosphate_first_24h_max
    → Electrolyte disturbances that affect weaning, arrhythmias, bone/renal status.

Blood Gas Dynamics
• ph_first24h / ph_last24h / ph_delta
    → First and last pH in first 24h, and change over time; acidosis correction or worsening.
• pco2_first24h / pco2_last24h / pco2_delta
    → Ventilation and respiratory compensation dynamics.

Availability & QC
• glucose_available_flag, creatinine_available_flag,
  lactate_available_flag, hemoglobin_available_flag
    → 1 if the corresponding feature family is present; useful for:
        – missingness modelling,
        – data-quality monitoring,
        – deciding imputation strategy.
• magnesium_fallback_nobs, phosphate_fallback_nobs, crp_fallback_nobs
    → Number of raw labevents used in fallback for these analytes (for audit).
• feature_extraction_ts
    → Timestamp when this feature row was created (lineage, reproducibility).

Usage for Readmission Modeling
------------------------------
- Direct numeric predictors:
    • all *_first_24h_min / *_max continuous features.
- Binary severity flags:
    • severe_anemia_flag, elevated_wbc_flag, acute_kidney_injury_flag,
      elevated_lactate_flag, low_albumin_flag, hypo/hyperglycemia flags.
- Combined with vitals, GCS, comorbidities, and interventions, these labs help
  the model capture **acute organ failure** patterns that strongly influence
  30-day ICU readmission risk.

==============================================================================*/


-- ============================================================================
-- TABLE BUILD: feature_labs_first_24h
-- ============================================================================

CREATE OR REPLACE TABLE `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h` AS

WITH
-- 1) COHORT: canonical index ICU stays
cohort AS (
  SELECT subject_id, hadm_id, index_stay_id, index_icu_intime, readmit_30d_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
),

-- 2) DERIVED: safely convert each derived.first_day_lab row to JSON then extract scalars
derived_json AS (
  SELECT
    SAFE_CAST(stay_id AS INT64) AS stay_id,
    TO_JSON_STRING(t) AS json_row
  FROM `physionet-data.mimiciv_3_1_derived.first_day_lab` AS t
),

derived_parsed AS (
  SELECT
    stay_id,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.hemoglobin_min') AS FLOAT64)        AS hemoglobin_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.hemoglobin_max') AS FLOAT64)        AS hemoglobin_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.hematocrit_min') AS FLOAT64)        AS hematocrit_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.hematocrit_max') AS FLOAT64)        AS hematocrit_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.platelets_min') AS FLOAT64)         AS platelets_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.platelets_max') AS FLOAT64)         AS platelets_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.wbc_max') AS FLOAT64)               AS wbc_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.glucose_min') AS FLOAT64)           AS glucose_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.glucose_max') AS FLOAT64)           AS glucose_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.creatinine_max') AS FLOAT64)        AS creatinine_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.bun_max') AS FLOAT64)               AS bun_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.sodium_min') AS FLOAT64)            AS sodium_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.sodium_max') AS FLOAT64)            AS sodium_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.potassium_min') AS FLOAT64)         AS potassium_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.potassium_max') AS FLOAT64)         AS potassium_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.chloride_min') AS FLOAT64)          AS chloride_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.bicarbonate_min') AS FLOAT64)       AS bicarbonate_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.calcium_min') AS FLOAT64)           AS calcium_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.aniongap_min') AS FLOAT64)          AS aniongap_min_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.albumin_min') AS FLOAT64)           AS albumin_min_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.bilirubin_total_max') AS FLOAT64)   AS bilirubin_total_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.bilirubin_direct_max') AS FLOAT64)  AS bilirubin_direct_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.alt_max') AS FLOAT64)               AS alt_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.ast_max') AS FLOAT64)               AS ast_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.inr_max') AS FLOAT64)               AS inr_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.pt_max') AS FLOAT64)                AS pt_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.ptt_max') AS FLOAT64)               AS ptt_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.d_dimer_max') AS FLOAT64)           AS d_dimer_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.fibrinogen_max') AS FLOAT64)        AS fibrinogen_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.lactate_max') AS FLOAT64)           AS lactate_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.troponin_max') AS FLOAT64)          AS troponin_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.ntprobnp_max') AS FLOAT64)          AS ntprobnp_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.abs_neutrophils_max') AS FLOAT64)   AS abs_neutrophils_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.abs_lymphocytes_max') AS FLOAT64)   AS abs_lymphocytes_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.abs_monocytes_max') AS FLOAT64)     AS abs_monocytes_max_drv,

    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.magnesium_max') AS FLOAT64)         AS magnesium_max_drv,
    SAFE_CAST(JSON_EXTRACT_SCALAR(json_row, '$.phosphate_max') AS FLOAT64)         AS phosphate_max_drv

  FROM derived_json
),

-- 3) LABEVENTS fallback: records within 0–24h of index ICU intime for selected itemids
labevents_24h AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    le.itemid,
    le.charttime,
    SAFE_CAST(le.valuenum AS FLOAT64) AS val
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_hosp.labevents` AS le
    ON le.subject_id = c.subject_id
   AND le.hadm_id    = c.hadm_id
   AND le.charttime >= c.index_icu_intime
   AND le.charttime < TIMESTAMP_ADD(c.index_icu_intime, INTERVAL 24 HOUR)
  WHERE le.valuenum IS NOT NULL
    AND le.itemid IN (
      50931, 50943, 50809,  -- glucose candidates
      50912,                -- creatinine
      51006,                -- BUN
      50813,                -- lactate
      51002, 50954, 51003,  -- troponin candidates
      50960,                -- magnesium
      50970,                -- phosphate
      50889,                -- CRP
      50820,                -- pH (blood)
      50818                 -- pCO2 (blood)
    )
),

-- 4) Aggregate per stay+itemid: min/max/mean + first/last values
labevents_agg AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    itemid,
    MIN(val) AS min_val,
    MAX(val) AS max_val,
    AVG(val) AS mean_val,
    COUNT(*) AS n_obs,
    ARRAY_AGG(STRUCT(charttime, val) ORDER BY charttime ASC LIMIT 1)[OFFSET(0)] AS first_row,
    ARRAY_AGG(STRUCT(charttime, val) ORDER BY charttime DESC LIMIT 1)[OFFSET(0)] AS last_row
  FROM labevents_24h
  GROUP BY subject_id, hadm_id, index_stay_id, itemid
),

-- 5) Pivot the aggregated labevents to wide columns (fallbacks)
labevents_pivot AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,

    -- Glucose
    MAX(CASE WHEN itemid IN (50931, 50943, 50809) THEN min_val END) AS glucose_min_le,
    MAX(CASE WHEN itemid IN (50931, 50943, 50809) THEN max_val END) AS glucose_max_le,

    -- Creatinine / BUN
    MAX(CASE WHEN itemid = 50912 THEN max_val END) AS creatinine_max_le,
    MAX(CASE WHEN itemid = 51006 THEN max_val END) AS bun_max_le,

    -- Lactate
    MAX(CASE WHEN itemid = 50813 THEN max_val END) AS lactate_max_le,

    -- Troponin
    MAX(CASE WHEN itemid IN (51002, 50954, 51003) THEN max_val END) AS troponin_max_le,

    -- Magnesium / Phosphate
    MAX(CASE WHEN itemid = 50960 THEN max_val END) AS magnesium_max_le,
    MAX(CASE WHEN itemid = 50970 THEN max_val END) AS phosphate_max_le,

    -- CRP
    MAX(CASE WHEN itemid = 50889 THEN max_val END) AS crp_max_le,

    -- Blood gases: first/last values
    MAX(CASE WHEN itemid = 50820 THEN first_row.val END) AS ph_first_le,
    MAX(CASE WHEN itemid = 50820 THEN last_row.val END)  AS ph_last_le,
    MAX(CASE WHEN itemid = 50818 THEN first_row.val END) AS pco2_first_le,
    MAX(CASE WHEN itemid = 50818 THEN last_row.val END)  AS pco2_last_le,

    -- fallback n_obs for QC
    MAX(CASE WHEN itemid = 50960 THEN n_obs END) AS mg_n_obs,
    MAX(CASE WHEN itemid = 50970 THEN n_obs END) AS phos_n_obs,
    MAX(CASE WHEN itemid = 50889 THEN n_obs END) AS crp_n_obs

  FROM labevents_agg
  GROUP BY subject_id, hadm_id, index_stay_id
),

-- 6) Final merge: derived_parsed (preferred) + labevents_pivot (fallback)
final AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    c.readmit_30d_flag,

    -- Hematology
    drv.hemoglobin_min_drv  AS hemoglobin_first_24h_min,
    drv.hemoglobin_max_drv  AS hemoglobin_first_24h_max,
    CASE
      WHEN drv.hemoglobin_min_drv IS NOT NULL AND drv.hemoglobin_min_drv < 7
      THEN 1 ELSE 0
    END AS severe_anemia_flag,

    drv.hematocrit_min_drv  AS hematocrit_first_24h_min,
    drv.hematocrit_max_drv  AS hematocrit_first_24h_max,

    drv.platelets_min_drv   AS platelets_first_24h_min,
    drv.platelets_max_drv   AS platelets_first_24h_max,

    drv.wbc_max_drv         AS wbc_first_24h_max,
    CASE
      WHEN drv.wbc_max_drv IS NOT NULL AND drv.wbc_max_drv > 15
      THEN 1 ELSE 0
    END AS elevated_wbc_flag,

    -- Glucose (derived → fallback)
    COALESCE(drv.glucose_min_drv, lvp.glucose_min_le) AS glucose_first_24h_min,
    COALESCE(drv.glucose_max_drv, lvp.glucose_max_le) AS glucose_first_24h_max,
    CASE
      WHEN COALESCE(drv.glucose_min_drv, lvp.glucose_min_le) IS NOT NULL
       AND COALESCE(drv.glucose_min_drv, lvp.glucose_min_le) < 70
      THEN 1 ELSE 0
    END AS hypoglycemia_flag,
    CASE
      WHEN COALESCE(drv.glucose_max_drv, lvp.glucose_max_le) IS NOT NULL
       AND COALESCE(drv.glucose_max_drv, lvp.glucose_max_le) > 180
      THEN 1 ELSE 0
    END AS hyperglycemia_flag,

    -- Renal function
    COALESCE(drv.creatinine_max_drv, lvp.creatinine_max_le) AS creatinine_first_24h_max,
    COALESCE(drv.bun_max_drv, lvp.bun_max_le)               AS bun_first_24h_max,
    CASE
      WHEN COALESCE(drv.creatinine_max_drv, lvp.creatinine_max_le) IS NOT NULL
       AND COALESCE(drv.creatinine_max_drv, lvp.creatinine_max_le) > 2.0
      THEN 1 ELSE 0
    END AS acute_kidney_injury_flag,

    -- Electrolytes & acid-base
    drv.sodium_min_drv        AS sodium_first_24h_min,
    drv.sodium_max_drv        AS sodium_first_24h_max,
    drv.potassium_min_drv     AS potassium_first_24h_min,
    drv.potassium_max_drv     AS potassium_first_24h_max,
    drv.chloride_min_drv      AS chloride_first_24h_min,
    drv.bicarbonate_min_drv   AS bicarbonate_first_24h_min,
    drv.calcium_min_drv       AS calcium_first_24h_min,
    drv.aniongap_min_drv      AS aniongap_first_24h_min,

    -- Liver
    drv.albumin_min_drv           AS albumin_first_24h_min,
    CASE
      WHEN drv.albumin_min_drv IS NOT NULL AND drv.albumin_min_drv < 2.0
      THEN 1 ELSE 0
    END AS low_albumin_flag,
    drv.bilirubin_total_max_drv   AS bilirubin_total_first_24h_max,
    drv.bilirubin_direct_max_drv  AS bilirubin_direct_first_24h_max,
    drv.alt_max_drv               AS alt_first_24h_max,
    drv.ast_max_drv               AS ast_first_24h_max,

    -- Coagulation
    drv.inr_max_drv        AS inr_first_24h_max,
    drv.pt_max_drv         AS pt_first_24h_max,
    drv.ptt_max_drv        AS ptt_first_24h_max,
    drv.d_dimer_max_drv    AS d_dimer_first_24h_max,
    drv.fibrinogen_max_drv AS fibrinogen_first_24h_max,

    -- Lactate (derived → fallback)
    COALESCE(drv.lactate_max_drv, lvp.lactate_max_le) AS lactate_first_24h_max,
    CASE
      WHEN COALESCE(drv.lactate_max_drv, lvp.lactate_max_le) IS NOT NULL
       AND COALESCE(drv.lactate_max_drv, lvp.lactate_max_le) > 4
      THEN 1 ELSE 0
    END AS elevated_lactate_flag,

    -- Cardiac markers
    COALESCE(drv.troponin_max_drv, lvp.troponin_max_le) AS troponin_first_24h_max,
    drv.ntprobnp_max_drv                                AS ntprobnp_first_24h_max,

    -- Differential
    drv.abs_neutrophils_max_drv AS abs_neutrophils_first_24h_max,
    drv.abs_lymphocytes_max_drv AS abs_lymphocytes_first_24h_max,
    drv.abs_monocytes_max_drv   AS abs_monocytes_first_24h_max,

    -- Magnesium & Phosphate (derived → fallback)
    COALESCE(drv.magnesium_max_drv, lvp.magnesium_max_le) AS magnesium_first_24h_max,
    COALESCE(drv.phosphate_max_drv, lvp.phosphate_max_le) AS phosphate_first_24h_max,

    -- Blood gas pH / pCO2 (fallback first/last)
    lvp.ph_first_le  AS ph_first24h,
    lvp.ph_last_le   AS ph_last24h,
    CASE
      WHEN lvp.ph_first_le IS NOT NULL AND lvp.ph_last_le IS NOT NULL
      THEN ROUND(lvp.ph_last_le - lvp.ph_first_le, 3)
      ELSE NULL
    END AS ph_delta,

    lvp.pco2_first_le AS pco2_first24h,
    lvp.pco2_last_le  AS pco2_last24h,
    CASE
      WHEN lvp.pco2_first_le IS NOT NULL AND lvp.pco2_last_le IS NOT NULL
      THEN ROUND(lvp.pco2_last_le - lvp.pco2_first_le, 2)
      ELSE NULL
    END AS pco2_delta,

    -- Availability flags
    CASE WHEN COALESCE(drv.glucose_min_drv,     lvp.glucose_min_le)     IS NOT NULL THEN 1 ELSE 0 END AS glucose_available_flag,
    CASE WHEN COALESCE(drv.creatinine_max_drv,  lvp.creatinine_max_le)  IS NOT NULL THEN 1 ELSE 0 END AS creatinine_available_flag,
    CASE WHEN COALESCE(drv.lactate_max_drv,     lvp.lactate_max_le)     IS NOT NULL THEN 1 ELSE 0 END AS lactate_available_flag,
    CASE WHEN drv.hemoglobin_min_drv IS NOT NULL THEN 1 ELSE 0 END AS hemoglobin_available_flag,

    -- fallback observation counts for audit
    lvp.mg_n_obs   AS magnesium_fallback_nobs,
    lvp.phos_n_obs AS phosphate_fallback_nobs,
    lvp.crp_n_obs  AS crp_fallback_nobs,

    CURRENT_TIMESTAMP() AS feature_extraction_ts

  FROM cohort c
  LEFT JOIN derived_parsed drv
    ON c.index_stay_id = drv.stay_id
  LEFT JOIN labevents_pivot lvp
    ON c.subject_id    = lvp.subject_id
   AND c.hadm_id       = lvp.hadm_id
   AND c.index_stay_id = lvp.index_stay_id
)

SELECT *
FROM final
ORDER BY subject_id, hadm_id;



-- ============================================================================
-- QC SUITE: feature_labs_first_24h
-- ============================================================================


-- QC 1: Row count alignment with cohort
--  - Ensures we have exactly one labs row per index ICU stay.
SELECT
  'QC1_RowCount' AS check_name,
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`) AS cohort_rows,
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`)      AS lab_rows,
  CASE
    WHEN (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`)
       = (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`)
    THEN '✅ 1:1 cohort → labs'
    ELSE '⚠️ mismatch, investigate joins'
  END AS status;



-- QC 2: Core lab coverage (how many rows have each major lab family)
--  - Helps us understand missingness and decide imputation strategy.
SELECT
  'QC2_Coverage' AS check_name,
  COUNT(*) AS total_rows,

  COUNTIF(glucose_available_flag   = 1) AS n_glucose,
  COUNTIF(creatinine_available_flag = 1) AS n_creatinine,
  COUNTIF(lactate_available_flag   = 1) AS n_lactate,
  COUNTIF(hemoglobin_available_flag = 1) AS n_hemoglobin,

  ROUND(100.0 * COUNTIF(glucose_available_flag   = 1) / COUNT(*), 1) AS pct_glucose,
  ROUND(100.0 * COUNTIF(creatinine_available_flag = 1) / COUNT(*), 1) AS pct_creatinine,
  ROUND(100.0 * COUNTIF(lactate_available_flag   = 1) / COUNT(*), 1) AS pct_lactate,
  ROUND(100.0 * COUNTIF(hemoglobin_available_flag = 1) / COUNT(*), 1) AS pct_hemoglobin
FROM `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`;



-- QC 3: Clinical range sanity checks (simple counts)
--  - Very loose bounds just to detect unit errors / parsing bugs.
SELECT
  'QC3_RangeSanity' AS check_name,
  COUNT(*) AS total_rows,

  -- Glucose (mg/dL)
  COUNTIF(glucose_first_24h_min IS NOT NULL AND (glucose_first_24h_min < 5 OR glucose_first_24h_min > 2000)) AS extreme_glucose_min,
  COUNTIF(glucose_first_24h_max IS NOT NULL AND (glucose_first_24h_max < 5 OR glucose_first_24h_max > 2000)) AS extreme_glucose_max,

  -- Creatinine (mg/dL)
  COUNTIF(creatinine_first_24h_max IS NOT NULL AND (creatinine_first_24h_max < 0.1 OR creatinine_first_24h_max > 20)) AS extreme_creatinine_max,

  -- Lactate (mmol/L)
  COUNTIF(lactate_first_24h_max IS NOT NULL AND (lactate_first_24h_max < 0.1 OR lactate_first_24h_max > 40)) AS extreme_lactate_max,

  -- Sodium (mEq/L)
  COUNTIF(sodium_first_24h_min IS NOT NULL AND (sodium_first_24h_min < 100 OR sodium_first_24h_min > 200)) AS extreme_sodium_min,
  COUNTIF(sodium_first_24h_max IS NOT NULL AND (sodium_first_24h_max < 100 OR sodium_first_24h_max > 200)) AS extreme_sodium_max,

  -- Potassium (mEq/L)
  COUNTIF(potassium_first_24h_min IS NOT NULL AND (potassium_first_24h_min < 1 OR potassium_first_24h_min > 12)) AS extreme_potassium_min,
  COUNTIF(potassium_first_24h_max IS NOT NULL AND (potassium_first_24h_max < 1 OR potassium_first_24h_max > 12)) AS extreme_potassium_max
FROM `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`;



-- QC 4: Distribution by readmission status
--  - Quick check that key labs show some separation between readmitted vs not.
SELECT
  'QC4_Distribution_By_Readmit' AS check_name,
  readmit_30d_flag,
  COUNT(*) AS n_rows,
  ROUND(AVG(glucose_first_24h_max), 1)      AS avg_glucose_max,
  ROUND(AVG(creatinine_first_24h_max), 2)   AS avg_creatinine_max,
  ROUND(AVG(lactate_first_24h_max), 2)      AS avg_lactate_max,
  ROUND(AVG(hemoglobin_first_24h_min), 2)   AS avg_hemoglobin_min,
  ROUND(AVG(albumin_first_24h_min), 2)      AS avg_albumin_min,
  ROUND(AVG(inr_first_24h_max), 2)          AS avg_inr_max
FROM `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`
GROUP BY readmit_30d_flag
ORDER BY readmit_30d_flag;





/*==============================================================================
PROJECT : ICU 30-Day Readmission – Neurological (GCS) Feature Layer
TABLE   : feature_neurological
AUTHOR  : Jyoti Prakash Das
VERSION : 1.0 (first-day GCS, cleaned & bucketed)
LAST UPDATED: 2025-12-10

PURPOSE
-------
We built this table to capture **neurological severity in the first 24 hours**
of the index ICU stay using the Glasgow Coma Scale (GCS).

Why this layer was important:
  - GCS is a core ICU severity signal (brain function, coma, sedation).
  - Early neurological status strongly influences:
      • mortality,
      • ICU length of stay,
      • readmission risk (difficult weaning, delirium, chronic impairment).
  - We needed a clean, ML-ready representation of:
      • total GCS score,
      • component subscores (eyes, verbal, motor),
      • severity buckets (severe/moderate/mild),
      • binary impairment flags,
      • availability and “unable to assess” flags.

DRIVER COHORT
-------------
nomadic-freedom-436306-g4.readmission30.feature_demographics_extended
  - One row per index ICU stay.
  - Keys used: subject_id, hadm_id, index_stay_id, readmit_30d_flag.

DATA SOURCES
------------
1) physionet-data.mimiciv_3_1_derived.first_day_gcs
   - stay_id-level, derived GCS features for first ICU day.
   - Columns used (after SAFE_CAST):
       • gcs_eyes, gcs_verbal, gcs_motor
       • gcs_min (min total in first day)
       • gcs_unable (indicator if exam was not possible)

2) feature_demographics_extended
   - Provides the canonical index_stay_id and readmit_30d_flag label.

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) Derived-only, no chartevents reconstruction
   - We relied on physionet’s first_day_gcs derived table (already ICU-day
     aligned and cleaned).
   - This avoided re-building GCS from raw flowsheets (which is messy with
     sedation, multiple scores, etc.).

2) Strict clinical ranges
   - We enforced valid GCS ranges:
       • Eyes   : 1–4
       • Verbal : 1–5
       • Motor  : 1–6
       • Total  : 3–15
   - Out-of-range values were treated as NULL to avoid corrupting severity.

3) Robust total score computation
   - We preferred gcs_min (lowest GCS in first day) when present.
   - If gcs_min was missing but all components were valid, we recomputed:
       gcs_total_first_24h = eyes + verbal + motor.
   - This ensured maximal usable coverage with a consistent “worst GCS” notion.

4) Severity bucketing and impairment flags
   - We encoded clinically familiar GCS buckets:
       • Severe   : total ≤ 8
       • Moderate : 9–12
       • Mild     : 13–14
       • Normal   : 15
   - We added targeted flags:
       • severe_gcs_depression_flag  (total ≤ 8)
       • severe_motor_response_flag  (motor ≤ 2)
       • severe_verbal_response_flag (verbal ≤ 2)
       • no_eye_opening_flag         (eyes = 1)
       • any_neuro_impairment_flag   (total < 15)

5) Availability and “unable to assess” handling
   - gcs_unable_to_assess_flag = 1 when first_day_gcs marked exam impossible.
   - gcs_complete_assessment_flag = 1 only if:
       • all three components present,
       • and gcs_unable_to_assess_flag = 0.
   - This makes it easy to:
       • analyze bias due to sedation/monitoring,
       • use missingness as a feature.

──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

Core Keys
• subject_id
    → Patient identifier; used to join all feature layers.
• hadm_id
    → Hospital admission identifier.
• index_stay_id
    → Index ICU stay identifier; aligns GCS with other first-24h features.
• readmit_30d_flag
    → 0/1 label: ICU readmission within 30 days (target variable).

Primary GCS Features
• gcs_total_first_24h
    → Final total GCS score for first ICU day (3–15).
      – We used gcs_min when available; otherwise sum of components.
      – Lower score = worse neuro status = potentially higher readmission risk.
• gcs_total_first_24h_min
    → Minimum total GCS from derived table (if provided; 3–15).
      – Represents “worst” neuro state in first 24h.

Component Subscores
• gcs_eyes_first_24h (1–4)
    → Eye opening response (spontaneous to none).
• gcs_verbal_first_24h (1–5)
    → Verbal response (oriented to none).
• gcs_motor_first_24h (1–6)
    → Motor response (obeys commands to none).

Assessment Limitations
• gcs_unable_to_assess_flag
    → 1 if GCS examination was flagged “unable to assess” (e.g., deep sedation,
      intubation without reliable score); 0 otherwise.

Severity & Impairment Flags
• severe_gcs_depression_flag
    → 1 if gcs_total_first_24h ≤ 8 (severe coma / neuro depression).
• moderate_gcs_flag
    → 1 if gcs_total_first_24h ∈ [9,12] (moderate impairment).
• mild_gcs_flag
    → 1 if gcs_total_first_24h ∈ [13,14] (mild impairment).
• any_neuro_impairment_flag
    → 1 if gcs_total_first_24h < 15 (any deviation from fully alert).
• severe_motor_response_flag
    → 1 if gcs_motor_first_24h ≤ 2 (decerebrate/decorticate/none).
• severe_verbal_response_flag
    → 1 if gcs_verbal_first_24h ≤ 2 (incomprehensible or none).
• no_eye_opening_flag
    → 1 if gcs_eyes_first_24h = 1 (no eye opening).

Availability Flags
• gcs_total_available_flag
    → 1 if gcs_total_first_24h is non-null; else 0.
• gcs_eyes_available_flag / gcs_verbal_available_flag / gcs_motor_available_flag
    → 1 if each component is present; else 0.
• gcs_complete_assessment_flag
    → 1 if:
        – all three components present
        – and gcs_unable_to_assess_flag = 0
      → indicates a reliable and complete GCS exam.

Metadata
• feature_extraction_timestamp
    → When this feature row was created (for lineage & reproducibility).

Usage for Readmission Modeling
------------------------------
- Direct numerical feature:
    • gcs_total_first_24h as a continuous severity score.
- Categorical features:
    • severity buckets (encoded via severe/moderate/mild flags).
- Binary risk signals:
    • any_neuro_impairment_flag, severe component flags, unable_to_assess.
- Combined with labs, vitals, comorbidities, these features capture **early
  neurological status**, which is strongly associated with ICU course and
  30-day readmission risk.

==============================================================================*/


-- ============================================================================
-- TABLE BUILD: feature_neurological
-- ============================================================================

CREATE OR REPLACE TABLE `nomadic-freedom-436306-g4.readmission30.feature_neurological` AS

-- STEP 1 — Cohort driver (one row per index ICU stay)
WITH cohort AS (
  SELECT
      subject_id,
      hadm_id,
      index_stay_id,
      readmit_30d_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
),

-- STEP 2 — Join first_day_gcs and SAFE_CAST values
gcs_raw AS (
  SELECT
      c.subject_id,
      c.hadm_id,
      c.index_stay_id,
      c.readmit_30d_flag,

      SAFE_CAST(fdg.gcs_eyes   AS INT64) AS gcs_eyes_raw,
      SAFE_CAST(fdg.gcs_verbal AS INT64) AS gcs_verbal_raw,
      SAFE_CAST(fdg.gcs_motor  AS INT64) AS gcs_motor_raw,
      SAFE_CAST(fdg.gcs_min    AS INT64) AS gcs_min_raw,
      SAFE_CAST(fdg.gcs_unable AS INT64) AS gcs_unable_raw

  FROM cohort c
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_gcs` fdg
    ON c.index_stay_id = fdg.stay_id
   AND c.subject_id    = fdg.subject_id
),

-- STEP 3 — Clean values using valid GCS clinical ranges
cleaned AS (
  SELECT
      subject_id,
      hadm_id,
      index_stay_id,
      readmit_30d_flag,

      CASE WHEN gcs_eyes_raw   BETWEEN 1 AND 4  THEN gcs_eyes_raw   END AS gcs_eyes_first_24h,
      CASE WHEN gcs_verbal_raw BETWEEN 1 AND 5  THEN gcs_verbal_raw END AS gcs_verbal_first_24h,
      CASE WHEN gcs_motor_raw  BETWEEN 1 AND 6  THEN gcs_motor_raw  END AS gcs_motor_first_24h,
      CASE WHEN gcs_min_raw    BETWEEN 3 AND 15 THEN gcs_min_raw    END AS gcs_total_first_24h_min,

      CASE WHEN gcs_unable_raw = 1 THEN 1 ELSE 0 END AS gcs_unable_to_assess_flag
  FROM gcs_raw
),

-- STEP 4 — Compute final total GCS (prefer min, else sum of components)
computed AS (
  SELECT
      subject_id,
      hadm_id,
      index_stay_id,
      readmit_30d_flag,
      gcs_eyes_first_24h,
      gcs_verbal_first_24h,
      gcs_motor_first_24h,
      gcs_total_first_24h_min,
      gcs_unable_to_assess_flag,

      COALESCE(
        gcs_total_first_24h_min,
        CASE WHEN gcs_eyes_first_24h   IS NOT NULL
               AND gcs_verbal_first_24h IS NOT NULL
               AND gcs_motor_first_24h  IS NOT NULL
             THEN gcs_eyes_first_24h
                + gcs_verbal_first_24h
                + gcs_motor_first_24h
        END
      ) AS gcs_total_first_24h
  FROM cleaned
),

-- STEP 5 — Add severity categories + flags + availability
flags AS (
  SELECT
      subject_id,
      hadm_id,
      index_stay_id,
      readmit_30d_flag,

      gcs_total_first_24h,
      gcs_total_first_24h_min,
      gcs_eyes_first_24h,
      gcs_verbal_first_24h,
      gcs_motor_first_24h,
      gcs_unable_to_assess_flag,

      -- Severity flags
      CASE WHEN gcs_total_first_24h IS NOT NULL AND gcs_total_first_24h <= 8  THEN 1 ELSE 0 END AS severe_gcs_depression_flag,
      CASE WHEN gcs_total_first_24h IS NOT NULL AND gcs_total_first_24h BETWEEN 9 AND 12  THEN 1 ELSE 0 END AS moderate_gcs_flag,
      CASE WHEN gcs_total_first_24h IS NOT NULL AND gcs_total_first_24h BETWEEN 13 AND 14 THEN 1 ELSE 0 END AS mild_gcs_flag,
      CASE WHEN gcs_total_first_24h IS NOT NULL AND gcs_total_first_24h < 15               THEN 1 ELSE 0 END AS any_neuro_impairment_flag,

      -- Component-based severe flags
      CASE WHEN gcs_motor_first_24h  IS NOT NULL AND gcs_motor_first_24h  <= 2 THEN 1 ELSE 0 END AS severe_motor_response_flag,
      CASE WHEN gcs_verbal_first_24h IS NOT NULL AND gcs_verbal_first_24h <= 2 THEN 1 ELSE 0 END AS severe_verbal_response_flag,
      CASE WHEN gcs_eyes_first_24h   IS NOT NULL AND gcs_eyes_first_24h   = 1 THEN 1 ELSE 0 END AS no_eye_opening_flag,

      -- Availability flags
      CASE WHEN gcs_total_first_24h IS NOT NULL THEN 1 ELSE 0 END AS gcs_total_available_flag,
      CASE WHEN gcs_eyes_first_24h  IS NOT NULL THEN 1 ELSE 0 END AS gcs_eyes_available_flag,
      CASE WHEN gcs_verbal_first_24h IS NOT NULL THEN 1 ELSE 0 END AS gcs_verbal_available_flag,
      CASE WHEN gcs_motor_first_24h  IS NOT NULL THEN 1 ELSE 0 END AS gcs_motor_available_flag,

      -- Complete, reliable assessment (all components present & not “unable”)
      CASE
        WHEN gcs_eyes_first_24h IS NOT NULL
         AND gcs_verbal_first_24h IS NOT NULL
         AND gcs_motor_first_24h  IS NOT NULL
         AND gcs_unable_to_assess_flag = 0
        THEN 1 ELSE 0
      END AS gcs_complete_assessment_flag

  FROM computed
)

-- STEP 6 — Final output
SELECT
  *,
  CURRENT_TIMESTAMP() AS feature_extraction_timestamp
FROM flags
ORDER BY subject_id, hadm_id;



-- ============================================================================
-- QC SUITE: feature_neurological
-- ============================================================================


-- QC 1: GCS data availability
--  - Checks coverage of total and component scores.
--  - Helps understand missingness before modelling.
SELECT
  'QC1_GCS_Availability' AS check_name,
  COUNT(*) AS total_rows,

  -- Any total GCS value
  COUNTIF(gcs_total_first_24h IS NOT NULL)     AS has_gcs_total,
  COUNTIF(gcs_total_first_24h_min IS NOT NULL) AS has_gcs_min,

  -- Components
  COUNTIF(gcs_eyes_first_24h   IS NOT NULL)    AS has_gcs_eyes,
  COUNTIF(gcs_verbal_first_24h IS NOT NULL)    AS has_gcs_verbal,
  COUNTIF(gcs_motor_first_24h  IS NOT NULL)    AS has_gcs_motor,

  -- Coverage percentages
  ROUND(100 * COUNTIF(gcs_total_first_24h IS NOT NULL)     / COUNT(*), 1) AS gcs_total_coverage_pct,
  ROUND(100 * COUNTIF(gcs_eyes_first_24h   IS NOT NULL)    / COUNT(*), 1) AS gcs_eyes_coverage_pct,
  ROUND(100 * COUNTIF(gcs_verbal_first_24h IS NOT NULL)    / COUNT(*), 1) AS gcs_verbal_coverage_pct,
  ROUND(100 * COUNTIF(gcs_motor_first_24h  IS NOT NULL)    / COUNT(*), 1) AS gcs_motor_coverage_pct,

  -- Simple status based on total GCS coverage
  CASE
    WHEN ROUND(100 * COUNTIF(gcs_total_first_24h IS NOT NULL) / COUNT(*), 1) >= 90
      THEN '✅ PASS (>=90% total GCS coverage)'
    WHEN ROUND(100 * COUNTIF(gcs_total_first_24h IS NOT NULL) / COUNT(*), 1) >= 70
      THEN '⚠️  MEDIUM COVERAGE (70–90%)'
    ELSE '❌ LOW COVERAGE (<70%)'
  END AS status
FROM `nomadic-freedom-436306-g4.readmission30.feature_neurological`;



-- QC 2: GCS range validation
--  - Ensures all stored values respect valid GCS ranges.
SELECT
  'QC2_GCS_Ranges' AS check_name,
  COUNT(*) AS total_rows,

  -- Total GCS must be between 3 and 15
  COUNTIF(gcs_total_first_24h IS NOT NULL
          AND (gcs_total_first_24h < 3 OR gcs_total_first_24h > 15)) AS out_of_range_total,

  -- Components: eyes 1–4, verbal 1–5, motor 1–6
  COUNTIF(gcs_eyes_first_24h   IS NOT NULL
          AND (gcs_eyes_first_24h < 1 OR gcs_eyes_first_24h > 4)) AS invalid_eyes,

  COUNTIF(gcs_verbal_first_24h IS NOT NULL
          AND (gcs_verbal_first_24h < 1 OR gcs_verbal_first_24h > 5)) AS invalid_verbal,

  COUNTIF(gcs_motor_first_24h  IS NOT NULL
          AND (gcs_motor_first_24h < 1 OR gcs_motor_first_24h > 6)) AS invalid_motor,

  CASE
    WHEN
      COUNTIF(gcs_total_first_24h IS NOT NULL
              AND (gcs_total_first_24h < 3 OR gcs_total_first_24h > 15)) = 0
      AND COUNTIF(gcs_eyes_first_24h   IS NOT NULL
                  AND (gcs_eyes_first_24h < 1 OR gcs_eyes_first_24h > 4)) = 0
      AND COUNTIF(gcs_verbal_first_24h IS NOT NULL
                  AND (gcs_verbal_first_24h < 1 OR gcs_verbal_first_24h > 5)) = 0
      AND COUNTIF(gcs_motor_first_24h  IS NOT NULL
                  AND (gcs_motor_first_24h < 1 OR gcs_motor_first_24h > 6)) = 0
    THEN '✅ PASS (All within expected ranges)'
    ELSE '⚠️  Check out-of-range GCS values'
  END AS status
FROM `nomadic-freedom-436306-g4.readmission30.feature_neurological`;



-- QC 3: Severity distribution
--  - Summarises how many patients fall into each GCS severity bucket.
--  - Useful to sanity-check cohort severity mix and class balance.
WITH severity AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    gcs_total_first_24h,
    severe_gcs_depression_flag,
    moderate_gcs_flag,
    mild_gcs_flag,
    any_neuro_impairment_flag,

    CASE
      WHEN gcs_total_first_24h IS NULL
        THEN 'Unknown'
      WHEN gcs_total_first_24h <= 8
        THEN 'Severe (<=8)'
      WHEN gcs_total_first_24h BETWEEN 9 AND 12
        THEN 'Moderate (9–12)'
      WHEN gcs_total_first_24h BETWEEN 13 AND 14
        THEN 'Mild (13–14)'
      WHEN gcs_total_first_24h = 15
        THEN 'Normal (15)'
      ELSE 'OutOfRange'
    END AS severity_bucket
  FROM `nomadic-freedom-436306-g4.readmission30.feature_neurological`
)

SELECT
  'QC3_Severity_Distribution' AS check_name,
  severity_bucket,
  COUNT(*) AS count,
  ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct,
  SUM(severe_gcs_depression_flag) AS severe_flag_count,
  SUM(moderate_gcs_flag)          AS moderate_flag_count,
  SUM(mild_gcs_flag)              AS mild_flag_count,
  SUM(any_neuro_impairment_flag)  AS any_impairment_flag_count
FROM severity
GROUP BY severity_bucket
ORDER BY count DESC;





-- TAble 6 creation pre columns name check to ensure our code running - 
-- =====================================================================
-- QUICK SCHEMA SNAPSHOTS FOR MEDS & INTERVENTION TABLES
-- Purpose: visually confirm which ID/time columns exist (subject_id,
--          hadm_id, stay_id, starttime, endtime, etc.)
-- Note: INFORMATION_SCHEMA is blocked for physionet-data, so we inspect
--       schemas by selecting 0 or 1 rows and looking at the result.
-- =====================================================================

-- 1) ACE inhibitors (hospital prescriptions)
SELECT
  'acei' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.acei`
LIMIT 1;

-- 2) ARBs (hospital prescriptions)
SELECT
  'arb' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.arb`
LIMIT 1;

-- 3) Antibiotics (hospital prescriptions + ICU stay_id)
SELECT
  'antibiotic' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.antibiotic`
LIMIT 1;

-- 4) Vasoactive agent summary (per ICU stay, per time segment)
SELECT
  'vasoactive_agent' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.vasoactive_agent`
LIMIT 1;

-- 5) Norepinephrine infusion (ICU inputevents-derived)
SELECT
  'norepinephrine' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.norepinephrine`
LIMIT 1;

-- 6) Dopamine infusion
SELECT
  'dopamine' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.dopamine`
LIMIT 1;

-- 7) Epinephrine infusion
SELECT
  'epinephrine' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.epinephrine`
LIMIT 1;

-- 8) Dobutamine infusion
SELECT
  'dobutamine' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.dobutamine`
LIMIT 1;

-- 9) Milrinone infusion
SELECT
  'milrinone' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.milrinone`
LIMIT 1;

-- 10) Ventilation episodes
SELECT
  'ventilation' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.ventilation`
LIMIT 1;

-- 11) RRT (renal replacement therapy) episodes
SELECT
  'rrt' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.rrt`
LIMIT 1;

-- 12) CRRT episodes
SELECT
  'crrt' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.crrt`
LIMIT 1;

-- 13) Invasive lines
SELECT
  'invasive_line' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.invasive_line`
LIMIT 1;

-- 14) Neuromuscular blockade
SELECT
  'neuroblock' AS table_name,
  *
FROM `physionet-data.mimiciv_3_1_derived.neuroblock`
LIMIT 1;




/*==============================================================================
PROJECT : ICU 30-Day Readmission – Medications & Interventions Layer
TABLE   : feature_medications_interventions
AUTHOR  : Jyoti Prakash Das
VERSION : 1.0 (first 24h, ICU-indexed, binary flags + intensity scores)
LAST UPDATED: 2025-12-10

PURPOSE
-------
We built this table to summarize **medication and organ-support intensity in the
FIRST 24 HOURS of the index ICU stay** for our readmission cohort.

Why this layer was important:
  - First-day treatment intensity (pressors, ventilation, RRT, invasive lines)
    captured how sick the patient was and how aggressively they were treated.
  - Chronic meds (ACEI/ARB) plus early antibiotics gave signal about underlying
    cardiovascular disease and infection at admission.
  - We needed a **simple, robust set of binary flags and composite scores**
    that:
      • aligned with index_stay_id and index_icu_intime,
      • were easy to interpret in a clinical discussion,
      • could be reused across models and dashboards.

DRIVER COHORT
-------------
nomadic-freedom-436306-g4.readmission30.feature_demographics_extended
  - One row per index ICU stay in our 30-day readmission cohort.
  - Keys used: subject_id, hadm_id, index_stay_id, index_icu_intime, readmit_30d_flag.

DATA SOURCES (Confirmed)
------------------------
Hospital-level meds (admission-wide, time-stamped):
  - physionet-data.mimiciv_3_1_derived.acei
      • subject_id, hadm_id, acei, starttime, stoptime
  - physionet-data.mimiciv_3_1_derived.arb
      • subject_id, hadm_id, arb, starttime, stoptime
  - physionet-data.mimiciv_3_1_derived.antibiotic
      • subject_id, hadm_id, stay_id, antibiotic, route, starttime, stoptime

ICU-level vasoactive agents & interventions (stay-level, time-stamped):
  - vasoactive_agent: stay_id, starttime, endtime, dopamine, epinephrine,
                      norepinephrine, phenylephrine, vasopressin,
                      dobutamine, milrinone
  - norepinephrine, dopamine, epinephrine, dobutamine, milrinone:
      • stay_id, vaso_rate, vaso_amount, starttime, endtime
  - ventilation: stay_id, starttime, endtime, ventilation_status
  - rrt: stay_id, charttime, dialysis_present, dialysis_active, dialysis_type
  - crrt: stay_id, charttime, system_active, clotted flags, etc.
  - invasive_line: stay_id, line_type, line_site, starttime, endtime
  - neuroblock: stay_id, orderid, drug_rate, drug_amount, starttime, endtime

WINDOW DEFINITION (FIRST 24 HOURS)
----------------------------------
For each index ICU stay, we defined:
  - icu_start  = index_icu_intime
  - icu_24h_end = index_icu_intime + 24 hours

We treated a medication / procedure as “in first 24h” if **ANY part of the
interval overlapped** this window:

  - Interval-based records (starttime, stoptime/endtime):
      • kept if   (starttime < icu_24h_end) AND (stoptime/endtime > icu_start)

  - Charttime-based records (rrt / crrt):
      • kept if   charttime in [icu_start, icu_24h_end)

KEY DESIGN DECISIONS / ISSUES WE SOLVED
---------------------------------------
1) ICU-indexed windowing
   - We drove everything off index_icu_intime (not hospital admit time).
   - This reflected ICU-level acuity during the first 24 hours, which was more
     relevant for ICU readmission risk.

2) Overlap logic instead of “during only”
   - We considered a drug/procedure “present in first 24h” if **any overlap**
     existed with the 0–24h window.
   - This avoided missing chronic infusions that started slightly before ICU
     admission but were still active during the first ICU day.

3) Granular flags + composite scores
   - We created clear boolean flags per treatment (e.g., norepinephrine_24h_flag).
   - We then derived composite scores:
       • medication_intensity_score_24h
       • treatment_intensity_score_24h
       • high_acuity_24h_flag (mech vent + vasopressor)
   - These made it easy to use either fine-grained features or higher-level
     severity abstractions.

4) “Chronic cardio meds” bundle
   - We combined ACEI and ARB into chronic_cardio_med_flag
     to capture underlying cardiovascular disease and guideline-based therapy.

5) ICU organ support abstraction
   - Mechanical ventilation, vasopressors, RRT/CRRT, and invasive lines were
     treated as **organ-support counts** to approximate treatment intensity.

──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

Core Keys & Label
• subject_id
    → Unique patient ID; used to join with other feature tables.
• hadm_id
    → Hospital admission ID; ties ICU treatment to admission context.
• index_stay_id
    → Index ICU stay ID; aligns with vitals, labs, GCS, etc.
• readmit_30d_flag
    → 1 if ICU readmission occurred within 30 days; our target label.

Medication Flags (First 24h)
• acei_24h_flag
    → 1 if an ACE inhibitor overlapped with the first 24h ICU window.
      Why: reflects chronic heart failure / hypertension management.
• arb_24h_flag
    → 1 if an ARB overlapped with the first 24h window.
• chronic_cardio_med_flag
    → 1 if either ACEI or ARB was active in the first 24h.
      Why: marker of chronic cardiovascular disease and outpatient care quality.
• antibiotic_24h_flag
    → 1 if any antibiotic course overlapped the first 24h ICU window.
      Why: indicates suspected or confirmed infection/sepsis at ICU admission.

Vasopressors & Inotropes (First 24h)
• vasopressor_24h_flag
    → 1 if any vasoactive_agent row overlapped first 24h (pressors/inotropes).
      Why: direct signal of shock/instability.
• norepinephrine_24h_flag / dopamine_24h_flag / epinephrine_24h_flag
    → 1 if that specific agent was used in the first 24h.
      Why: different agents imply different shock phenotypes & severity.
• dobutamine_24h_flag / milrinone_24h_flag
    → 1 if inotrope infusion was present.
      Why: reflects low cardiac output / cardiogenic shock.
• any_inotrope_24h_flag
    → 1 if dobutamine or milrinone was used.
      Why: simple “cardiac inotropy” marker.

Organ Support & Procedures (First 24h)
• mechanical_ventilation_24h_flag
    → 1 if patient received mechanical ventilation during first 24h.
      Why: key marker of respiratory failure and ICU acuity.
• rrt_24h_flag
    → 1 if RRT (dialysis) charttime occurred in first 24h.
• crrt_24h_flag
    → 1 if CRRT activity occurred in first 24h.
      Why: both indicate acute kidney injury / multi-organ failure.
• invasive_line_24h_flag
    → 1 if any invasive line (e.g., central line, arterial line) was placed or
      active in first 24h.
      Why: proxy for severity and monitoring needs.
• neuroblock_24h_flag
    → 1 if neuromuscular blockade was used in first 24h.
      Why: reflects deep sedation, severe ARDS, or high acuity ventilation.

Composite Scores
• medication_intensity_score_24h
    → acei_24h_flag + arb_24h_flag + vasopressor_24h_flag + antibiotic_24h_flag  
      Why: quick summary of “how many key drug categories” were active.
• treatment_intensity_score_24h
    → mechanical_ventilation_24h_flag + vasopressor_24h_flag +
      rrt_24h_flag + invasive_line_24h_flag  
      Why: count of major organ-support modalities; higher = sicker.
• high_acuity_24h_flag
    → 1 if mechanical_ventilation_24h_flag = 1 AND vasopressor_24h_flag = 1.  
      Why: classic pattern of shock + respiratory failure; very high risk.

Metadata
• medications_data_available
    → 1 (table was built for all cohort stays); placeholder to keep contract.
• feature_extraction_timestamp
    → Timestamp when this feature row was generated; used for lineage and
      reproducibility across runs.

Usage for Readmission Modeling
------------------------------
- Direct binary predictors:
    • each flag (e.g., vasopressor_24h_flag, rrt_24h_flag) as independent
      risk factors.
- Composite severity features:
    • treatment_intensity_score_24h and high_acuity_24h_flag as strong
      early acuity markers.
- Interaction with comorbidities and demographics:
    • helps distinguish chronically sick but stable vs acutely unstable
      patients, which is crucial for 30-day readmission risk profiling.

==============================================================================*/


-- ============================================================================
-- TABLE BUILD: feature_medications_interventions
-- ============================================================================

CREATE OR REPLACE TABLE `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions` AS

WITH
-- 0) Cohort driver: one row per index ICU stay
cohort AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    index_icu_intime,
    TIMESTAMP_ADD(index_icu_intime, INTERVAL 24 HOUR) AS icu_24h_end,
    readmit_30d_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
),

-- 1) ACEI in first 24h of ICU (admission-level, time-filtered)
acei_24h AS (
  SELECT DISTINCT
    c.subject_id,
    c.hadm_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.acei` a
    ON c.subject_id = a.subject_id
   AND c.hadm_id   = a.hadm_id
  WHERE a.starttime IS NOT NULL
    AND a.stoptime  IS NOT NULL
    AND a.starttime < c.icu_24h_end
    AND a.stoptime  > c.index_icu_intime
),

-- 2) ARB in first 24h of ICU
arb_24h AS (
  SELECT DISTINCT
    c.subject_id,
    c.hadm_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.arb` ar
    ON c.subject_id = ar.subject_id
   AND c.hadm_id   = ar.hadm_id
  WHERE ar.starttime IS NOT NULL
    AND ar.stoptime  IS NOT NULL
    AND ar.starttime < c.icu_24h_end
    AND ar.stoptime  > c.index_icu_intime
),

-- 3) Antibiotic in first 24h of ICU
antibiotic_24h AS (
  SELECT DISTINCT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.antibiotic` ab
    ON c.subject_id = ab.subject_id
   AND c.hadm_id   = ab.hadm_id
   AND (ab.stay_id IS NULL OR SAFE_CAST(ab.stay_id AS INT64) = c.index_stay_id)
  WHERE ab.starttime IS NOT NULL
    AND ab.stoptime  IS NOT NULL
    AND ab.starttime < c.icu_24h_end
    AND ab.stoptime  > c.index_icu_intime
),

-- 4) Any vasoactive_agent in first 24h
vaso_24h AS (
  SELECT DISTINCT
    c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.vasoactive_agent` v
    ON v.stay_id = c.index_stay_id
  WHERE v.starttime IS NOT NULL
    AND v.endtime   IS NOT NULL
    AND v.starttime < c.icu_24h_end
    AND v.endtime   > c.index_icu_intime
),

-- 5) Specific pressor / inotrope tables (presence only)
norepi_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.norepinephrine` n
    ON n.stay_id = c.index_stay_id
  WHERE n.starttime IS NOT NULL
    AND n.endtime   IS NOT NULL
    AND n.starttime < c.icu_24h_end
    AND n.endtime   > c.index_icu_intime
),

dopamine_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.dopamine` d
    ON d.stay_id = c.index_stay_id
  WHERE d.starttime IS NOT NULL
    AND d.endtime   IS NOT NULL
    AND d.starttime < c.icu_24h_end
    AND d.endtime   > c.index_icu_intime
),

epi_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.epinephrine` e
    ON e.stay_id = c.index_stay_id
  WHERE e.starttime IS NOT NULL
    AND e.endtime   IS NOT NULL
    AND e.starttime < c.icu_24h_end
    AND e.endtime   > c.index_icu_intime
),

dobu_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.dobutamine` db
    ON db.stay_id = c.index_stay_id
  WHERE db.starttime IS NOT NULL
    AND db.endtime   IS NOT NULL
    AND db.starttime < c.icu_24h_end
    AND db.endtime   > c.index_icu_intime
),

milri_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.milrinone` ml
    ON ml.stay_id = c.index_stay_id
  WHERE ml.starttime IS NOT NULL
    AND ml.endtime   IS NOT NULL
    AND ml.starttime < c.icu_24h_end
    AND ml.endtime   > c.index_icu_intime
),

-- 6) Ventilation episodes in first 24h
vent_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.ventilation` vt
    ON vt.stay_id = c.index_stay_id
  WHERE vt.starttime IS NOT NULL
    AND vt.endtime   IS NOT NULL
    AND vt.starttime < c.icu_24h_end
    AND vt.endtime   > c.index_icu_intime
),

-- 7) RRT & CRRT during first 24h (charttime-based)
rrt_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.rrt` r
    ON r.stay_id = c.index_stay_id
  WHERE r.charttime IS NOT NULL
    AND r.charttime >= c.index_icu_intime
    AND r.charttime <  c.icu_24h_end
    AND (r.dialysis_present = 1 OR r.dialysis_active = 1)
),

crrt_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.crrt` cr
    ON cr.stay_id = c.index_stay_id
  WHERE cr.charttime IS NOT NULL
    AND cr.charttime >= c.index_icu_intime
    AND cr.charttime <  c.icu_24h_end
),

-- 8) Invasive lines in first 24h
lines_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.invasive_line` l
    ON l.stay_id = c.index_stay_id
  WHERE l.starttime IS NOT NULL
    AND l.endtime   IS NOT NULL
    AND l.starttime < c.icu_24h_end
    AND l.endtime   > c.index_icu_intime
),

-- 9) Neuromuscular blockade in first 24h
neuroblock_24h AS (
  SELECT DISTINCT c.index_stay_id
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.neuroblock` nb
    ON nb.stay_id = c.index_stay_id
  WHERE nb.starttime IS NOT NULL
    AND nb.endtime   IS NOT NULL
    AND nb.starttime < c.icu_24h_end
    AND nb.endtime   > c.index_icu_intime
),

-- 10) Aggregate everything into one feature row per cohort stay
med_flags AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.index_stay_id,
    c.readmit_30d_flag,

    -- Medications (24h)
    CASE WHEN a.subject_id IS NOT NULL THEN 1 ELSE 0 END AS acei_24h_flag,
    CASE WHEN r.subject_id IS NOT NULL THEN 1 ELSE 0 END AS arb_24h_flag,
    CASE WHEN ab.subject_id IS NOT NULL THEN 1 ELSE 0 END AS antibiotic_24h_flag,

    CASE
      WHEN (a.subject_id IS NOT NULL OR r.subject_id IS NOT NULL) THEN 1
      ELSE 0
    END AS chronic_cardio_med_flag,

    -- Vasoactives / Inotropes (24h)
    CASE WHEN v.index_stay_id  IS NOT NULL THEN 1 ELSE 0 END AS vasopressor_24h_flag,
    CASE WHEN ne.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS norepinephrine_24h_flag,
    CASE WHEN dp.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS dopamine_24h_flag,
    CASE WHEN ep.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS epinephrine_24h_flag,
    CASE WHEN db.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS dobutamine_24h_flag,
    CASE WHEN ml.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS milrinone_24h_flag,

    CASE
      WHEN db.index_stay_id IS NOT NULL OR ml.index_stay_id IS NOT NULL THEN 1
      ELSE 0
    END AS any_inotrope_24h_flag,

    -- Organ support / procedures (24h)
    CASE WHEN vt.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS mechanical_ventilation_24h_flag,
    CASE WHEN rr.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS rrt_24h_flag,
    CASE WHEN cr.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS crrt_24h_flag,
    CASE WHEN ln.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS invasive_line_24h_flag,
    CASE WHEN nb.index_stay_id IS NOT NULL THEN 1 ELSE 0 END AS neuroblock_24h_flag

  FROM cohort c
  LEFT JOIN acei_24h       a  ON c.subject_id    = a.subject_id AND c.hadm_id = a.hadm_id
  LEFT JOIN arb_24h        r  ON c.subject_id    = r.subject_id AND c.hadm_id = r.hadm_id
  LEFT JOIN antibiotic_24h ab ON c.subject_id    = ab.subject_id
                              AND c.hadm_id      = ab.hadm_id
                              AND c.index_stay_id = ab.index_stay_id

  LEFT JOIN vaso_24h       v  ON c.index_stay_id = v.index_stay_id
  LEFT JOIN norepi_24h     ne ON c.index_stay_id = ne.index_stay_id
  LEFT JOIN dopamine_24h   dp ON c.index_stay_id = dp.index_stay_id
  LEFT JOIN epi_24h        ep ON c.index_stay_id = ep.index_stay_id
  LEFT JOIN dobu_24h       db ON c.index_stay_id = db.index_stay_id
  LEFT JOIN milri_24h      ml ON c.index_stay_id = ml.index_stay_id

  LEFT JOIN vent_24h       vt ON c.index_stay_id = vt.index_stay_id
  LEFT JOIN rrt_24h        rr ON c.index_stay_id = rr.index_stay_id
  LEFT JOIN crrt_24h       cr ON c.index_stay_id = cr.index_stay_id
  LEFT JOIN lines_24h      ln ON c.index_stay_id = ln.index_stay_id
  LEFT JOIN neuroblock_24h nb ON c.index_stay_id = nb.index_stay_id
)

-- 11) Final SELECT: composite scores + metadata
SELECT
  subject_id,
  hadm_id,
  index_stay_id,
  readmit_30d_flag,

  acei_24h_flag,
  arb_24h_flag,
  chronic_cardio_med_flag,
  antibiotic_24h_flag,

  vasopressor_24h_flag,
  norepinephrine_24h_flag,
  dopamine_24h_flag,
  epinephrine_24h_flag,
  dobutamine_24h_flag,
  milrinone_24h_flag,
  any_inotrope_24h_flag,

  mechanical_ventilation_24h_flag,
  rrt_24h_flag,
  crrt_24h_flag,
  invasive_line_24h_flag,
  neuroblock_24h_flag,

  (acei_24h_flag
   + arb_24h_flag
   + vasopressor_24h_flag
   + antibiotic_24h_flag) AS medication_intensity_score_24h,

  (mechanical_ventilation_24h_flag
   + vasopressor_24h_flag
   + rrt_24h_flag
   + invasive_line_24h_flag) AS treatment_intensity_score_24h,

  CASE
    WHEN mechanical_ventilation_24h_flag = 1 AND vasopressor_24h_flag = 1 THEN 1
    ELSE 0
  END AS high_acuity_24h_flag,

  1 AS medications_data_available,

  CURRENT_TIMESTAMP() AS feature_extraction_timestamp

FROM med_flags
ORDER BY subject_id, hadm_id;



-- ============================================================================
-- QC SUITE: feature_medications_interventions
-- ============================================================================

-- QC 1: Medication & procedure coverage in first 24h
--  - How often each treatment was used.
SELECT
  'QC1_Med_Coverage_24h' AS check_name,
  COUNT(*) AS total_rows,

  -- Medications
  SUM(acei_24h_flag)       AS acei_count,
  SUM(arb_24h_flag)        AS arb_count,
  SUM(antibiotic_24h_flag) AS antibiotic_count,

  -- Vaso / inotropes
  SUM(vasopressor_24h_flag)     AS vasopressor_count,
  SUM(norepinephrine_24h_flag)  AS norepi_count,
  SUM(dopamine_24h_flag)        AS dopamine_count,
  SUM(epinephrine_24h_flag)     AS epi_count,
  SUM(dobutamine_24h_flag)      AS dobu_count,
  SUM(milrinone_24h_flag)       AS milri_count,
  SUM(any_inotrope_24h_flag)    AS any_inotrope_count,

  -- Organ support / procedures
  SUM(mechanical_ventilation_24h_flag) AS vent_count,
  SUM(rrt_24h_flag)                    AS rrt_count,
  SUM(crrt_24h_flag)                   AS crrt_count,
  SUM(invasive_line_24h_flag)          AS line_count,
  SUM(neuroblock_24h_flag)             AS neuroblock_count,

  -- Key percentages (quick sanity checks)
  ROUND(100.0 * SUM(mechanical_ventilation_24h_flag) / COUNT(*), 1) AS vent_pct,
  ROUND(100.0 * SUM(vasopressor_24h_flag)          / COUNT(*), 1)   AS vasopressor_pct,
  ROUND(100.0 * SUM(antibiotic_24h_flag)           / COUNT(*), 1)   AS antibiotic_pct

FROM `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions`;


-- QC 2: Treatment intensity distribution
--  - How many organ supports per patient (0–4).
SELECT
  'QC2_Treatment_Intensity_24h' AS check_name,
  treatment_intensity_score_24h,
  COUNT(*) AS count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions`
GROUP BY treatment_intensity_score_24h
ORDER BY treatment_intensity_score_24h DESC;


-- QC 3: High-acuity cases (vent + vasopressor)
--  - Proportion of sickest patients in our cohort.
SELECT
  'QC3_High_Acuity_24h' AS check_name,
  COUNT(*) AS total_rows,
  SUM(high_acuity_24h_flag) AS high_acuity_count,
  ROUND(100.0 * SUM(high_acuity_24h_flag) / COUNT(*), 1) AS high_acuity_pct,

  SUM(CASE WHEN treatment_intensity_score_24h >= 3 THEN 1 ELSE 0 END) AS high_intensity_count,
  ROUND(
    100.0 * SUM(CASE WHEN treatment_intensity_score_24h >= 3 THEN 1 ELSE 0 END) / COUNT(*),
    1
  ) AS high_intensity_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions`;


-- QC 4: Medication intensity distribution
--  - How many “aggressive treatment categories” per patient in first 24h.
SELECT
  'QC4_Med_Intensity_24h' AS check_name,
  medication_intensity_score_24h,
  COUNT(*) AS count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions`
GROUP BY medication_intensity_score_24h
ORDER BY medication_intensity_score_24h DESC;


-- QC 5: Relationship between treatment intensity and 30-day readmission
--  - Quick sanity check: does higher treatment_intensity_score_24h roughly
--    correspond to higher readmission rates?
SELECT
  'QC5_Intensity_vs_Readmission_24h' AS check_name,
  treatment_intensity_score_24h,
  COUNT(*) AS total_rows,
  SUM(readmit_30d_flag) AS readmit_count,
  ROUND(100.0 * SUM(readmit_30d_flag) / COUNT(*), 1) AS readmit_rate_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions`
GROUP BY treatment_intensity_score_24h
ORDER BY treatment_intensity_score_24h DESC;




-- Precheck Table column names already present to create Comorbidity Mapping Layer

-- Look at first few rows and columns of each table
SELECT * FROM `physionet-data.mimiciv_3_1_derived.charlson` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.sofa` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.apsiii` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.sapsii` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.lods` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.oasis` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.sepsis3` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.sirs` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.suspicion_of_infection` LIMIT 1;



/*==============================================================================
PROJECT : ICU 30-Day Readmission – ICD → Comorbidity Mapping Layer
TABLE   : feature_comorbidities_mapping_flags
AUTHOR  : Jyoti Prakash Das
VERSION : 1.0 (mapping-driven Charlson + Non-Charlson expansions)
LAST UPDATED: 2025-12-10

PURPOSE
-------
We created this layer to **convert raw ICD diagnosis codes into comorbidity
flags using a compact and editable mapping table**, instead of relying solely
on pre-computed Charlson tables.

Why this mapping table matters:
  ✓ Hospitals rarely provide pre-labeled Charlson fields in real EHR dumps  
  ✓ ICD ranges differ across institutions — we maintain a **clean & auditable map**  
  ✓ Allows **future expansion** (new ICD prefixes, pediatrics, local coding)  
  ✓ Supports **explainability** in ML models ("why was patient marked COPD?")

This table builds **binary comorbidity flags per admission**, which later feed
into:
  ● Charlson score computation  
  ● Severity views such as **feature_comorbidities_severity**  
  ● Model-ready static risk indicators  

We intentionally **included non-Charlson clinical predictors** like AFib and
Hypertension — useful for readmission prediction even if not part of Charlson.

INPUT TABLES
------------
1) mapping reference table: `readmission30.icd_comorbidity_map`
2) diagnoses table:        `mimiciv_3_1_hosp.diagnoses_icd`
3) cohort driver table:    `readmission30.mimiciv_index_cohort_30d`

OUTPUT TABLE
------------
nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags

OUTPUT COLUMNS & Why They Matter
--------------------------------
Core Identifiers
• subject_id, hadm_id → admission-level comorbidity flags

Comorbidity Flags (binary 0/1)
• myocardial_infarction_flag      → CAD risk, mortality risk, readmission
• chf_flag                        → major readmission predictor
• peripheral_vascular_flag        → vascular burden → postop & wound risk
• cerebrovascular_flag            → stroke history → rehab, complications
• dementia_flag                   → long-LOS, discharge complexity
• copd_flag                       → respiratory instability → high revisit rate
• diabetes_no_cc_flag             → metabolic baseline
• diabetes_with_cc_flag           → neuropathy/renal/cardiac → high-risk
• ckd_flag                        → dialysis probability, drug handling
• cancer_flag                     → recent chemotherapy/tracking care
• metastatic_tumor_flag           → very high acuity & mortality
• mild_liver_flag                 → risk of variceal bleed, HE
• severe_liver_flag               → coagulopathy, ICU mortality ++
• paraplegia_flag                 → infection/UTI risk, pressure ulcers
• aids_flag                       → immunocompromise — infection readmission
• afib_flag                       → non-Charlson but strong real-world predictor
• hypertension_flag               → chronic baseline stressor

Metadata
• mapping_extraction_ts → reproducibility, lineage tracking

==============================================================================*/


-- ===========================================================================
-- STEP A — ICD → COMORBIDITY MAPPING REFERENCE TABLE
-- Compact, editable, transparent lookup for ICD9/ICD10 → disease groups
-- ===========================================================================
CREATE OR REPLACE TABLE `nomadic-freedom-436306-g4.readmission30.icd_comorbidity_map` AS
SELECT * FROM UNNEST([
  -- disease_key,      disease_name,                 icd_version, code_prefix, charlson_weight
  STRUCT('myocardial_infarction'    AS disease_key, 'Acute MI'                      AS disease_name, 10 AS icd_version, 'I21'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('myocardial_infarction'    AS disease_key, 'Acute MI (ICD9)'               AS disease_name,  9 AS icd_version, '410'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('congestive_heart_failure' AS disease_key, 'CHF'                           AS disease_name, 10 AS icd_version, 'I50'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('congestive_heart_failure' AS disease_key, 'CHF (ICD9)'                    AS disease_name,  9 AS icd_version, '428'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('peripheral_vascular'      AS disease_key, 'Peripheral vascular disease'   AS disease_name, 10 AS icd_version, 'I70'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('peripheral_vascular'      AS disease_key, 'Peripheral vascular (ICD9)'    AS disease_name,  9 AS icd_version, '440'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('cerebrovascular'          AS disease_key, 'Stroke/CVA'                    AS disease_name, 10 AS icd_version, 'I60'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('cerebrovascular'          AS disease_key, 'Stroke (ICD9)'                 AS disease_name,  9 AS icd_version, '430'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('dementia'                 AS disease_key, 'Dementia'                      AS disease_name, 10 AS icd_version, 'F00'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('dementia'                 AS disease_key, 'Dementia (ICD9)'               AS disease_name,  9 AS icd_version, '290'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('chronic_pulmonary'        AS disease_key, 'COPD'                          AS disease_name, 10 AS icd_version, 'J44'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('chronic_pulmonary'        AS disease_key, 'COPD (ICD9)'                   AS disease_name,  9 AS icd_version, '490'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('rheumatic_disease'        AS disease_key, 'Rheumatic'                     AS disease_name, 10 AS icd_version, 'M05'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('peptic_ulcer'             AS disease_key, 'Peptic ulcer'                  AS disease_name, 10 AS icd_version, 'K25'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('mild_liver'               AS disease_key, 'Mild liver disease'            AS disease_name, 10 AS icd_version, 'K73'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('mild_liver'               AS disease_key, 'Mild liver (ICD9)'             AS disease_name,  9 AS icd_version, '571'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('diabetes_no_cc'           AS disease_key, 'Diabetes w/o CC'               AS disease_name, 10 AS icd_version, 'E10'   AS code_prefix, 1 AS charlson_weight),
  STRUCT('diabetes_with_cc'         AS disease_key, 'Diabetes with CC'              AS disease_name, 10 AS icd_version, 'E11'   AS code_prefix, 2 AS charlson_weight),
  STRUCT('diabetes_no_cc'           AS disease_key, 'Diabetes (ICD9)'               AS disease_name,  9 AS icd_version, '250'   AS code_prefix, 1 AS charlson_weight),

  STRUCT('paraplegia'               AS disease_key, 'Paraplegia'                    AS disease_name, 10 AS icd_version, 'G81'   AS code_prefix, 2 AS charlson_weight),
  STRUCT('renal_disease'            AS disease_key, 'CKD'                           AS disease_name, 10 AS icd_version, 'N18'   AS code_prefix, 2 AS charlson_weight),

  STRUCT('malignant_cancer'         AS disease_key, 'Cancer'                        AS disease_name, 10 AS icd_version, 'C00'   AS code_prefix, 2 AS charlson_weight),
  STRUCT('metastatic_tumor'         AS disease_key, 'Metastatic tumor'              AS disease_name, 10 AS icd_version, 'C77'   AS code_prefix, 6 AS charlson_weight),

  STRUCT('severe_liver'             AS disease_key, 'Severe liver disease'          AS disease_name, 10 AS icd_version, 'K72'   AS code_prefix, 3 AS charlson_weight),
  STRUCT('aids'                     AS disease_key, 'HIV/AIDS'                      AS disease_name, 10 AS icd_version, 'B20'   AS code_prefix, 6 AS charlson_weight),

  -- Non-Charlson but highly predictive
  STRUCT('afib'                     AS disease_key, 'AFib/Flutter'                  AS disease_name, 10 AS icd_version, 'I48'   AS code_prefix, 0 AS charlson_weight),
  STRUCT('afib'                     AS disease_key, 'AFib (ICD9)'                   AS disease_name,  9 AS icd_version, '42731' AS code_prefix, 0 AS charlson_weight),
  STRUCT('hypertension'             AS disease_key, 'Hypertension'                  AS disease_name, 10 AS icd_version, 'I10'   AS code_prefix, 0 AS charlson_weight)
]);

-- ===========================================================================
-- STEP B — Create flags per admission based on ICD matches
-- ===========================================================================
CREATE OR REPLACE TABLE `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags` AS
WITH cohort AS (
  SELECT subject_id, hadm_id
  FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`
),

diag_filtered AS (
  SELECT d.hadm_id, d.icd_version, d.icd_code
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  INNER JOIN cohort c USING(hadm_id)
  WHERE d.icd_code IS NOT NULL
),

diag_mapped AS (
  SELECT
    df.hadm_id,
    m.disease_key,
    m.charlson_weight,
    df.icd_code
  FROM diag_filtered df
  JOIN `nomadic-freedom-436306-g4.readmission30.icd_comorbidity_map` m
    ON m.icd_version = df.icd_version
   AND df.icd_code LIKE CONCAT(m.code_prefix, '%')
)

SELECT
  c.subject_id,
  c.hadm_id,
  MAX(CASE WHEN disease_key = 'myocardial_infarction'    THEN 1 ELSE 0 END) AS myocardial_infarction_flag,
  MAX(CASE WHEN disease_key = 'congestive_heart_failure' THEN 1 ELSE 0 END) AS chf_flag,
  MAX(CASE WHEN disease_key = 'peripheral_vascular'      THEN 1 ELSE 0 END) AS peripheral_vascular_flag,
  MAX(CASE WHEN disease_key = 'cerebrovascular'          THEN 1 ELSE 0 END) AS cerebrovascular_flag,
  MAX(CASE WHEN disease_key = 'dementia'                 THEN 1 ELSE 0 END) AS dementia_flag,
  MAX(CASE WHEN disease_key = 'chronic_pulmonary'        THEN 1 ELSE 0 END) AS copd_flag,
  MAX(CASE WHEN disease_key = 'diabetes_no_cc'           THEN 1 ELSE 0 END) AS diabetes_no_cc_flag,
  MAX(CASE WHEN disease_key = 'diabetes_with_cc'         THEN 1 ELSE 0 END) AS diabetes_with_cc_flag,
  MAX(CASE WHEN disease_key = 'renal_disease'            THEN 1 ELSE 0 END) AS ckd_flag,
  MAX(CASE WHEN disease_key = 'malignant_cancer'         THEN 1 ELSE 0 END) AS cancer_flag,
  MAX(CASE WHEN disease_key = 'metastatic_tumor'         THEN 1 ELSE 0 END) AS metastatic_tumor_flag,
  MAX(CASE WHEN disease_key = 'mild_liver'               THEN 1 ELSE 0 END) AS mild_liver_flag,
  MAX(CASE WHEN disease_key = 'severe_liver'             THEN 1 ELSE 0 END) AS severe_liver_flag,
  MAX(CASE WHEN disease_key = 'paraplegia'               THEN 1 ELSE 0 END) AS paraplegia_flag,
  MAX(CASE WHEN disease_key = 'aids'                     THEN 1 ELSE 0 END) AS aids_flag,
  MAX(CASE WHEN disease_key = 'afib'                     THEN 1 ELSE 0 END) AS afib_flag,
  MAX(CASE WHEN disease_key = 'hypertension'             THEN 1 ELSE 0 END) AS hypertension_flag,
  CURRENT_TIMESTAMP() AS mapping_extraction_ts
FROM cohort c
LEFT JOIN diag_mapped dm USING(hadm_id)
GROUP BY c.subject_id, c.hadm_id;





-- ===========================================================================
-- QC BLOCK (copy-paste ready)
-- ===========================================================================
-- QC1: Coverage check
SELECT
  'QC1_Comorbidity_Flag_Distribution' AS check_name,
  COUNT(*) AS total_rows,
  ROUND(100 * AVG(copd_flag),1) AS pct_copd,
  ROUND(100 * AVG(chf_flag),1) AS pct_chf,
  ROUND(100 * AVG(diabetes_with_cc_flag),1) AS pct_diabetes_cc,
  ROUND(100 * AVG(ckd_flag),1) AS pct_ckd,
  ROUND(100 * AVG(cancer_flag),1) AS pct_cancer,
  ROUND(100 * AVG(afib_flag),1) AS pct_afib
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags`;


/*==============================================================================
ICD → COMORBIDITY MAPPING: BACKING SOURCES & VALIDATION(For Above Table)
==============================================================================

Why Above Comorbidty Mapping table exists 
---------------------------------------
We needed a **small, explicit mapping layer** from raw ICD-9/ICD-10 diagnosis
codes to high-level comorbidity buckets (e.g., myocardial_infarction,
chronic_pulmonary, diabetes_with_cc). This mapping table powered:

  - feature_comorbidities_mapping_flags  (binary 0/1 flags per hadm_id)
  - charlson-style severity scoring and additional AFib/HTN features
  - downstream views like feature_comorbidities_severity

Instead of inline CASE/regex scattered in SQL, we centralized all prefixes
in `icd_comorbidity_map` using simple LIKE prefix rules (e.g. icd_code LIKE 'I50%').

Primary scientific sources (Charlson-style mappings)
----------------------------------------------------
The ICD prefixes and Charlson weights we used were aligned with standard,
peer-reviewed coding algorithms for administrative data:

1) Quan H, Sundararajan V, Halfon P, et al.
   "Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10
    administrative data."
   Medical Care. 2005;43(11):1130-1139.
   DOI: 10.1097/01.mlr.0000182534.19832.83
   (Widely used reference for Charlson & Elixhauser ICD-9/10 mappings)
   Link (example copy of algorithms PDF):
   - https://mchp-appserv.cpe.umanitoba.ca/concept/Elixhauser%20Comorbidities%20-%20Coding%20Algorithms%20for%20ICD-9-CM%20and%20ICD-10.pdf

2) Deyo RA, Cherkin DC, Ciol MA.
   "Adapting a clinical comorbidity index for use with ICD-9-CM
    administrative databases."
   J Clin Epidemiol. 1992;45(6):613-619.
   DOI: 10.1016/0895-4356(92)90133-8
   (Classic ICD-9 adaptation of the Charlson Comorbidity Index)
   - https://doi.org/10.1016/0895-4356(92)90133-8

We based the **disease list and weights** on Charlson conventions from these
papers (e.g., diabetes_with_cc weight=2, metastatic_tumor weight=6,
severe_liver weight=3, etc.) and then encoded them as simple
`disease_key` + `charlson_weight` columns.

Clinical code validation
------------------------
For each ICD prefix we added (e.g. 'I21', 'I50', 'E10', 'E11', 'N18',
'C00', 'C77', 'K73', 'K72', 'B20', etc.), we verified clinical meaning
against official / widely used ICD references:

  • Official WHO ICD-10 Browser  
    - https://icd.who.int/browse10/  
    Used to confirm that each prefix family corresponds to the intended
    comorbidity:
      - I21* → Acute myocardial infarction  
      - I50* → Heart failure  
      - J44* → Other chronic obstructive pulmonary disease  
      - E10*, E11* → Diabetes mellitus categories  
      - N18* → Chronic kidney disease  
      - C00*-C97* / C77* → Malignant / metastatic neoplasms  
      - B20* → HIV disease  
      - etc.

  • ICD10Data.com (convenient lookup; not an official standard but widely used)  
    - https://www.icd10data.com/  
    Used as a secondary check on descriptions and inclusion of specific
    subcodes (e.g., I48* for atrial fibrillation, I10* for essential
    hypertension).

Logic and limitations
---------------------
- We intentionally used **coarse prefixes** (e.g. 'I21', 'I50', 'J44', 'N18')
  so they covered the main Charlson-relevant ranges described in Quan/Deyo.
- This mapping was **simpler** than the full Quan algorithm:
    • It did not enumerate every individual subcode (e.g. I50.2, I50.3, …),
      but relied on prefix ranges that match all child codes.
    • It did not implement every exclusion code from the full published
      algorithms (e.g., to avoid double-counting certain malignancies).
- For production clinical research, any grey-area prefix can be audited by:
    1) Looking up the code in WHO ICD-10 Browser / ICD10Data.
    2) Comparing it with the published Quan supplemental code lists.
    3) Updating `icd_comorbidity_map` (add/remove rows) with a new
       deployment and re-running the mapping pipeline.

Bottom line (correctness stance)
--------------------------------
- The **disease families and weights** match standard Charlson conventions
  (per Quan 2005 and Deyo 1992).
- The **exact prefix granularity** is a **conscious simplification** to keep
  the mapping table small, auditable, and practical for real-world deployment.
- If a site/hospital has its own validated Charlson code lists, they can
  be imported directly into `icd_comorbidity_map` to replace or extend
  these prefixes without changing the downstream SQL.

==============================================================================*/






-- FREE STORAGE QUOTA USED UP HAVE TO CLEAN THE OLD DASETS or From here create VIEW instead of Table

-- ===================================================================
-- 0) CONFIRM: tables you want to KEEP (no SQL here, just comments)
--    KEEP:
--      charlson_score
--      feature_anthropometry
--      feature_comorbidities_mapping_flags
--      feature_demographics_extended
--      feature_labs_first_24h
--      feature_medications_interventions
--      feature_neurological
--      feature_vitals_first_24h
--      icd_comorbidity_map
--      mimiciv_index_cohort_30d
-- ===================================================================


-- ===================================================================
-- 1) Pre-check: show sizes & row counts for ALL tables in dataset
-- ===================================================================
SELECT
  table_id AS table_name,
  row_count,
  ROUND(size_bytes/1024/1024,2) AS size_mb,
  ROUND(size_bytes/1024/1024/1024,4) AS size_gb,
  TIMESTAMP_MILLIS(last_modified_time) AS last_modified_ts
FROM `nomadic-freedom-436306-g4.readmission30.__TABLES__`
ORDER BY size_bytes DESC;


-- ===================================================================
-- 1b) Pre-check: only candidate tables you might DROP
-- ===================================================================
SELECT
  table_id AS table_name,
  row_count,
  ROUND(size_bytes/1024/1024,2) AS size_mb,
  TIMESTAMP_MILLIS(last_modified_time) AS last_modified_ts
FROM `nomadic-freedom-436306-g4.readmission30.__TABLES__`
WHERE table_id IN (
  'Corrected_Itemid_Mapping',
  'mapping_itemid',
  'metre_events',
  'metre_summary_24h',
  'vitals_qc_outliers',
  'final_model_dataset'       -- OPTIONAL: review before deleting
)
ORDER BY size_bytes DESC;

-- ================================
-- DROP large unused tables safely
-- ================================
DROP TABLE IF EXISTS `nomadic-freedom-436306-g4.readmission30.metre_events`;
DROP TABLE IF EXISTS `nomadic-freedom-436306-g4.readmission30.metre_summary_24h`;
DROP TABLE IF EXISTS `nomadic-freedom-436306-g4.readmission30.final_model_dataset`; -- optional, can re-create later

-- ================================
-- Check remaining storage after drop
-- ================================
SELECT
  table_id AS table_name,
  row_count,
  ROUND(size_bytes/1024/1024,2) AS size_mb,
  ROUND(size_bytes/1024/1024/1024,4) AS size_gb,
  TIMESTAMP_MILLIS(last_modified_time) AS last_modified_ts
FROM `nomadic-freedom-436306-g4.readmission30.__TABLES__`
ORDER BY size_bytes DESC;



-- Total storage used in this dataset (MB + GB)
SELECT
  SUM(size_bytes)/1024/1024 AS total_mb,
  SUM(size_bytes)/1024/1024/1024 AS total_gb
FROM `nomadic-freedom-436306-g4.readmission30.__TABLES__`;




SELECT
  table_id,
  ROUND(size_bytes/1024/1024,2) AS size_MB,
  ROUND(size_bytes/1024/1024/1024,3) AS size_GB
FROM `nomadic-freedom-436306-g4.readmission30.__TABLES__`
ORDER BY size_bytes DESC;


-- OK for some reason , we hvae hit storage quota limit in sql , so from now i will be creating and using VIEW instead of table (so feature table 7 and 8 are in VIEW ) :-

/* ============================================================================
   VIEW  : feature_comorbidities_severity
   PATH  : nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity
   GRAIN : 1 row per index hospital admission (mimiciv_index_cohort_30d)

   PURPOSE
   -------
   A unified severity + comorbidity layer that merges:
     1) Chronic comorbidity burden:
          • Charlson index components (disease history)
          • ICD-based mapping flags (AFib, Hypertension, Cancer, etc.)

     2) Acute first-24h severity:
          • SOFA, APSIII, SAPSII, LODS, OASIS (multi-organ failure scoring)
          • KDIGO stage (acute kidney injury grade)

     3) Sepsis/Inflammation/Infection indicators:
          • Sepsis-3, SIRS, suspicion_of_infection

   Why this feature matters:
     • Readmission risk depends on BOTH chronic disease burden + acute severity.
     • This view gives a single place to pull *clinical baseline + acute physiology*
       at admission for model training, phenotyping, dashboards.


   DATA SOURCES / INPUT TABLES
   ---------------------------
   COHORT DRIVER
     • mimiciv_index_cohort_30d
       (subject_id, hadm_id, index_stay_id, readmit_30d_flag)

   CHRONIC COMORBIDITY
     • physionet-data.mimiciv_3_1_derived.charlson
     • feature_comorbidities_mapping_flags   (custom ICD mapping table)

   ACUTE SEVERITY (FIRST 24H)
     • physionet-data.mimiciv_3_1_derived.sofa
     • physionet-data.mimiciv_3_1_derived.apsiii
     • physionet-data.mimiciv_3_1_derived.sapsii
     • physionet-data.mimiciv_3_1_derived.lods
     • physionet-data.mimiciv_3_1_derived.oasis
     • physionet-data.mimiciv_3_1_derived.kdigo_stages

   INFECTION / SEPSIS
     • physionet-data.mimiciv_3_1_derived.sepsis3
     • physionet-data.mimiciv_3_1_derived.sirs
     • physionet-data.mimiciv_3_1_derived.suspicion_of_infection


   KEY DESIGN DECISIONS / ISSUES WE SOLVED
   ---------------------------------------
   1) Standardised chronic disease layer
      • Used Charlson (gold standard) for classical comorbidities.
      • Added our ICD mapping flags for AFib/HTN/cancer to improve recall.

   2) ICU severity = first 24 hours only
      • All score tables are aggregated to **1 row per stay_id** using MAX().
      • Avoids row explosion and reduces compute.

   3) Sparse/partial availability handled cleanly
      • No hard filtering: missing scores remain NULL.
      • Added *_available_flag columns to capture missingness as signal.

   4) Central “one-stop feature layer”
      • Downstream ML pipelines can join this view alone for severity & comorbidity,
        instead of joining 8+ tables each time.

   5) Risk buckets for interpretability
      • sofa_risk_category defined for clinical communication & dashboards.


──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

IDENTIFIERS & LABEL
-------------------
• subject_id / hadm_id / index_stay_id  
    → Join keys across feature tables.

• readmit_30d_flag  
    → Model target (1 = readmitted within 30 days).


CHRONIC COMORBIDITY — CHARLSON COMPONENTS
-----------------------------------------
(Each column is binary unless numeric score)

• myocardial_infarct / CHF / COPD / CKD / cancer / etc.  
    → Encodes presence of specific chronic disease                         |
      Why important: chronic burden drives utilisation & post-discharge risk.

• charlson_comorbidity_index  
    → Total weighted score of comorbidities.                              |
      Higher index → higher mortality + readmission probability.


EXTRA ICD MAPPING FLAGS (More granular risk)
--------------------------------------------
• afib_flag / hypertension_flag / cancer_flag / etc.  
    → Captures comorbidities **not included in Charlson**.  
      Useful predictors for elderly, cardiac, oncology cohorts.

• total_comorbidity_count_mapping  
    → Simple additive burden measure.  
      Handy quick feature for XGBoost/Logistic models.


ACUTE SEVERITY (FIRST 24H SCORES)
---------------------------------
• sofa_score_first_24h, apsiii_score_first_24h, sapsii_score_first_24h  
    → Global severity in initial ICU stabilization window.  
      Early instability strongly predicts bounce-back admissions.

• lods_score_first_24h, oasis_score_first_24h  
    → Alternate illness severity scaling — complementary to SOFA.

• kdigo_stage_max_first_24h  
    → AKI grade (0–3).  
      AKI → prolonged recoveries, high post-discharge complication rate.


INFECTION & SEPSIS RISK
-----------------------
• sepsis3_flag  
    → 1 if meets Sepsis-3 criteria (infection + organ dysfunction).

• sirs_flag  
    → 1 if inflammatory response ≥2.

• suspicion_of_infection_flag + suspected_infection_time  
    → Captures clinicians' intent & workup timing — strong early signal.


INTERPRETABLE RISK BUCKET
-------------------------
• sofa_risk_category = Low / Moderate / High / Very High / Unknown  
    → Good summary for dashboards & feature summarization.


AVAILABILITY FLAGS
------------------
• charlson_available_flag / sofa_available_flag / ...  
    → Indicates missingness patterns important for ML and QA.


METADATA
--------
• feature_extraction_ts  
    → Timestamp for reproducibility & auditability.


──────────────────────────────────────────────────────────────────────────────
USAGE FOR READMISSION MODELING
──────────────────────────────────────────────────────────────────────────────

Recommended direct input features:
- charlson_comorbidity_index  
- total_comorbidity_count_mapping  
- sofa_score_first_24h, apsiii_score_first_24h, sapsii_score_first_24h  
- kdigo_stage_max_first_24h  
- sepsis3_flag, sirs_flag, suspicion_of_infection_flag  
- comorbidity binary indicators

Good categorical/meta features:
- sofa_risk_category  
- availability flags

Great for cohort analysis & interpretability:
- stratify readmission risk by chronic burden vs acute severity  
- investigate shock sepsis AKI clusters in high readmission population  
============================================================================== */


CREATE OR REPLACE VIEW
  `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity` AS

WITH cohort AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    readmit_30d_flag
  FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`
),

/* -----------------------------------------------------------------------------
   1) Audited Charlson (one row per hadm_id)
      Source: physionet-data.mimiciv_3_1_derived.charlson
      - We already QC'd that this table fully covers our cohort.
      - If we later prefer our own charlson_merged, we can swap the FROM.
----------------------------------------------------------------------------- */
charlson_small AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(myocardial_infarct           AS INT64) AS myocardial_infarct,
    SAFE_CAST(congestive_heart_failure     AS INT64) AS congestive_heart_failure,
    SAFE_CAST(peripheral_vascular_disease  AS INT64) AS peripheral_vascular_disease,
    SAFE_CAST(cerebrovascular_disease      AS INT64) AS cerebrovascular_disease,
    SAFE_CAST(dementia                     AS INT64) AS dementia,
    SAFE_CAST(chronic_pulmonary_disease    AS INT64) AS chronic_pulmonary_disease,
    SAFE_CAST(rheumatic_disease            AS INT64) AS rheumatic_disease,
    SAFE_CAST(peptic_ulcer_disease         AS INT64) AS peptic_ulcer_disease,
    SAFE_CAST(mild_liver_disease           AS INT64) AS mild_liver_disease,
    SAFE_CAST(diabetes_without_cc          AS INT64) AS diabetes_without_cc,
    SAFE_CAST(diabetes_with_cc             AS INT64) AS diabetes_with_cc,
    SAFE_CAST(paraplegia                   AS INT64) AS paraplegia,
    SAFE_CAST(renal_disease                AS INT64) AS renal_disease,
    SAFE_CAST(malignant_cancer             AS INT64) AS malignant_cancer,
    SAFE_CAST(severe_liver_disease         AS INT64) AS severe_liver_disease,
    SAFE_CAST(metastatic_solid_tumor       AS INT64) AS metastatic_solid_tumor,
    SAFE_CAST(aids                         AS INT64) AS aids,
    SAFE_CAST(charlson_comorbidity_index   AS INT64) AS charlson_comorbidity_index
  FROM `physionet-data.mimiciv_3_1_derived.charlson`
  WHERE hadm_id IN (SELECT hadm_id FROM cohort)
),

/* -----------------------------------------------------------------------------
   2) ICD mapping flags (our custom mapping table)
      - Already 1 row per hadm_id
      - Captures AFib, HTN, etc., which are useful but not in classic Charlson.
----------------------------------------------------------------------------- */
mapping_flags AS (
  SELECT *
  FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_mapping_flags`
  WHERE hadm_id IN (SELECT hadm_id FROM cohort)
),

/* -----------------------------------------------------------------------------
   3) Pre-aggregate severity scores — 1 row per stay_id
----------------------------------------------------------------------------- */

sofa_small AS (
  SELECT
    stay_id,
    SAFE_CAST(MAX(sofa_24hours)        AS INT64) AS sofa_score_first_24h,
    SAFE_CAST(MAX(respiration_24hours) AS INT64) AS sofa_respiration_24h,
    SAFE_CAST(MAX(coagulation_24hours) AS INT64) AS sofa_coagulation_24h,
    SAFE_CAST(MAX(liver_24hours)       AS INT64) AS sofa_liver_24h,
    SAFE_CAST(MAX(cardiovascular_24hours) AS INT64) AS sofa_cardiovascular_24h,
    SAFE_CAST(MAX(cns_24hours)         AS INT64) AS sofa_cns_24h,
    SAFE_CAST(MAX(renal_24hours)       AS INT64) AS sofa_renal_24h
  FROM `physionet-data.mimiciv_3_1_derived.sofa`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

apsiii_small AS (
  SELECT
    stay_id,
    SAFE_CAST(MAX(apsiii) AS INT64) AS apsiii_score_first_24h
  FROM `physionet-data.mimiciv_3_1_derived.apsiii`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

sapsii_small AS (
  SELECT
    stay_id,
    SAFE_CAST(MAX(sapsii) AS INT64) AS sapsii_score_first_24h
  FROM `physionet-data.mimiciv_3_1_derived.sapsii`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

lods_small AS (
  SELECT
    stay_id,
    SAFE_CAST(MAX(lods) AS INT64) AS lods_score_first_24h
  FROM `physionet-data.mimiciv_3_1_derived.lods`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

oasis_small AS (
  SELECT
    stay_id,
    SAFE_CAST(MAX(oasis) AS INT64) AS oasis_score_first_24h
  FROM `physionet-data.mimiciv_3_1_derived.oasis`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

kdigo_small AS (
  -- Worst (max) AKI stage per stay
  SELECT
    stay_id,
    SAFE_CAST(MAX(aki_stage_smoothed) AS INT64) AS kdigo_stage_max_first_24h
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

/* -----------------------------------------------------------------------------
   4) Sepsis / SIRS / Suspicion-of-infection (1 row per stay_id)
----------------------------------------------------------------------------- */
sepsis3_small AS (
  SELECT
    stay_id,
    CASE
      WHEN MAX(CASE WHEN sepsis3 THEN 1 ELSE 0 END) > 0 THEN 1
      ELSE 0
    END AS sepsis3_flag
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

sirs_small AS (
  SELECT
    stay_id,
    CASE
      WHEN MAX(IFNULL(SAFE_CAST(sirs AS INT64), 0)) >= 2 THEN 1
      ELSE 0
    END AS sirs_flag
  FROM `physionet-data.mimiciv_3_1_derived.sirs`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
),

soi_small AS (
  -- suspicion_of_infection can have multiple events / times per stay
  SELECT
    stay_id,
    ANY_VALUE(hadm_id) AS hadm_id,
    CASE
      WHEN MAX(CASE WHEN suspected_infection_time IS NOT NULL THEN 1 ELSE 0 END) > 0
        THEN 1 ELSE 0
    END AS suspected_infection_flag,
    MIN(suspected_infection_time) AS first_suspected_infection_time
  FROM `physionet-data.mimiciv_3_1_derived.suspicion_of_infection`
  WHERE stay_id IN (SELECT index_stay_id FROM cohort)
  GROUP BY stay_id
)

/* -----------------------------------------------------------------------------
   5) FINAL ASSEMBLY — 1 row per cohort hadm_id + index_stay_id
----------------------------------------------------------------------------- */
SELECT
  c.subject_id,
  c.hadm_id,
  c.index_stay_id,
  c.readmit_30d_flag,

  -- --------------- CHRONIC COMORBIDITIES (Charlson) ---------------
  ch.myocardial_infarct,
  ch.congestive_heart_failure,
  ch.peripheral_vascular_disease,
  ch.cerebrovascular_disease,
  ch.dementia,
  ch.chronic_pulmonary_disease,
  ch.rheumatic_disease,
  ch.peptic_ulcer_disease,
  ch.mild_liver_disease,
  ch.diabetes_without_cc,
  ch.diabetes_with_cc,
  ch.paraplegia,
  ch.renal_disease,
  ch.malignant_cancer,
  ch.severe_liver_disease,
  ch.metastatic_solid_tumor,
  ch.aids,
  ch.charlson_comorbidity_index,

  -- --------------- EXTRA ICD MAPPING FLAGS (custom map) ---------------
  mf.myocardial_infarction_flag,
  mf.chf_flag,
  mf.peripheral_vascular_flag,
  mf.cerebrovascular_flag,
  mf.dementia_flag,
  mf.copd_flag,
  mf.diabetes_no_cc_flag,
  mf.diabetes_with_cc_flag,
  mf.ckd_flag,
  mf.cancer_flag,
  mf.metastatic_tumor_flag,
  mf.mild_liver_flag,
  mf.severe_liver_flag,
  mf.paraplegia_flag,
  mf.aids_flag,
  mf.afib_flag,
  mf.hypertension_flag,

  -- Simple mapping-based “chronic burden” counter (tunable)
  (
    COALESCE(mf.chf_flag,0) +
    COALESCE(mf.copd_flag,0) +
    COALESCE(mf.ckd_flag,0) +
    COALESCE(mf.cancer_flag,0) +
    COALESCE(mf.afib_flag,0) +
    COALESCE(mf.hypertension_flag,0)
  ) AS total_comorbidity_count_mapping,

  -- --------------- ACUTE SEVERITY SCORES (first 24h) ---------------
  ss.sofa_score_first_24h,
  ss.sofa_respiration_24h,
  ss.sofa_coagulation_24h,
  ss.sofa_liver_24h,
  ss.sofa_cardiovascular_24h,
  ss.sofa_cns_24h,
  ss.sofa_renal_24h,

  ap.apsiii_score_first_24h,
  sa.sapsii_score_first_24h,
  ld.lods_score_first_24h,
  oa.oasis_score_first_24h,
  kd.kdigo_stage_max_first_24h,

  -- --------------- SEPSIS / INFECTION FLAGS ---------------
  sp.sepsis3_flag,
  sr.sirs_flag,
  COALESCE(soi.suspected_infection_flag, 0) AS suspicion_of_infection_flag,
  soi.first_suspected_infection_time        AS suspected_infection_time,

  -- --------------- SOFA RISK CATEGORY (example bins) ---------------
  CASE
    WHEN ss.sofa_score_first_24h IS NULL THEN 'Unknown'
    WHEN ss.sofa_score_first_24h >= 10   THEN 'Very High'
    WHEN ss.sofa_score_first_24h >= 7    THEN 'High'
    WHEN ss.sofa_score_first_24h >= 4    THEN 'Moderate'
    ELSE 'Low'
  END AS sofa_risk_category,

  -- --------------- AVAILABILITY FLAGS ---------------
  CASE WHEN ch.charlson_comorbidity_index IS NOT NULL THEN 1 ELSE 0 END AS charlson_available_flag,
  CASE WHEN ss.sofa_score_first_24h       IS NOT NULL THEN 1 ELSE 0 END AS sofa_available_flag,
  CASE WHEN ap.apsiii_score_first_24h     IS NOT NULL THEN 1 ELSE 0 END AS apsiii_available_flag,
  CASE WHEN sa.sapsii_score_first_24h     IS NOT NULL THEN 1 ELSE 0 END AS sapsii_available_flag,
  CASE WHEN kd.kdigo_stage_max_first_24h  IS NOT NULL THEN 1 ELSE 0 END AS kdigo_available_flag,

  CURRENT_TIMESTAMP() AS feature_extraction_ts

FROM cohort c
LEFT JOIN charlson_small ch
  ON c.hadm_id = ch.hadm_id
LEFT JOIN mapping_flags mf
  ON c.hadm_id = mf.hadm_id

LEFT JOIN sofa_small   ss ON c.index_stay_id = ss.stay_id
LEFT JOIN apsiii_small ap ON c.index_stay_id = ap.stay_id
LEFT JOIN sapsii_small sa ON c.index_stay_id = sa.stay_id
LEFT JOIN lods_small   ld ON c.index_stay_id = ld.stay_id
LEFT JOIN oasis_small  oa ON c.index_stay_id = oa.stay_id
LEFT JOIN kdigo_small  kd ON c.index_stay_id = kd.stay_id

LEFT JOIN sepsis3_small sp ON c.index_stay_id = sp.stay_id
LEFT JOIN sirs_small    sr ON c.index_stay_id = sr.stay_id
LEFT JOIN soi_small     soi ON c.index_stay_id = soi.stay_id

ORDER BY c.subject_id, c.hadm_id;


-- ============================================================================
-- QC1: Row Count & Uniqueness
-- Purpose:
--   • Check that the view has exactly one row per cohort hadm_id.
--   • Confirm row count == cohort row count.
-- ============================================================================

SELECT
  'QC1_RowCount_And_Uniqueness' AS check_name,
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`)
    AS cohort_rows,
  COUNT(*) AS view_rows,
  COUNT(DISTINCT hadm_id) AS distinct_hadm_ids,
  COUNT(*) - COUNT(DISTINCT hadm_id) AS duplicate_hadm_rows
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`;


-- Show any duplicated hadm_ids (should be zero rows ideally)
SELECT
  'QC1b_Duplicated_HADM' AS check_name,
  hadm_id,
  COUNT(*) AS cnt
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`
GROUP BY hadm_id
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 20;



-- ============================================================================
-- QC2: Availability Coverage of Key Scores
-- Purpose:
--   • Measure % coverage for Charlson and major severity scores.
--   • Helps decide which scores are usable as core model features.
-- ============================================================================

SELECT
  'QC2_Score_Availability' AS check_name,
  COUNT(*) AS total_rows,

  -- Charlson
  COUNTIF(charlson_comorbidity_index IS NOT NULL) AS n_charlson,
  ROUND(100.0 * COUNTIF(charlson_comorbidity_index IS NOT NULL) / COUNT(*), 1)
    AS pct_charlson,

  -- Severity scores
  COUNTIF(sofa_score_first_24h IS NOT NULL)  AS n_sofa,
  COUNTIF(apsiii_score_first_24h IS NOT NULL) AS n_apsiii,
  COUNTIF(sapsii_score_first_24h IS NOT NULL) AS n_sapsii,
  COUNTIF(lods_score_first_24h IS NOT NULL)   AS n_lods,
  COUNTIF(oasis_score_first_24h IS NOT NULL)  AS n_oasis,
  COUNTIF(kdigo_stage_max_first_24h IS NOT NULL) AS n_kdigo,

  ROUND(100.0 * COUNTIF(sofa_score_first_24h IS NOT NULL)  / COUNT(*), 1) AS pct_sofa,
  ROUND(100.0 * COUNTIF(apsiii_score_first_24h IS NOT NULL) / COUNT(*), 1) AS pct_apsiii,
  ROUND(100.0 * COUNTIF(sapsii_score_first_24h IS NOT NULL) / COUNT(*), 1) AS pct_sapsii,
  ROUND(100.0 * COUNTIF(lods_score_first_24h IS NOT NULL)   / COUNT(*), 1) AS pct_lods,
  ROUND(100.0 * COUNTIF(oasis_score_first_24h IS NOT NULL)  / COUNT(*), 1) AS pct_oasis,
  ROUND(100.0 * COUNTIF(kdigo_stage_max_first_24h IS NOT NULL) / COUNT(*), 1) AS pct_kdigo
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`;



-- ============================================================================
-- QC3: Distribution of Charlson & SOFA (sanity ranges)
-- Purpose:
--   • Quick distribution stats to confirm realistic ranges.
--   • SOFA and Charlson tails should look clinically reasonable.
-- ============================================================================

SELECT
  'QC3_Distribution_Charlson_Sofa' AS check_name,
  COUNT(*) AS total_rows,

  -- Charlson index
  ROUND(AVG(charlson_comorbidity_index), 2) AS avg_charlson,
  MIN(charlson_comorbidity_index)           AS min_charlson,
  MAX(charlson_comorbidity_index)           AS max_charlson,

  -- SOFA
  ROUND(AVG(sofa_score_first_24h), 2) AS avg_sofa,
  MIN(sofa_score_first_24h)           AS min_sofa,
  MAX(sofa_score_first_24h)           AS max_sofa
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`;



-- ============================================================================
-- QC4: Sepsis / SIRS / Suspicion-of-infection Prevalence
-- Purpose:
--   • Understand how many stays are flagged as septic / infected.
--   • Good for debugging suspicion_of_infection join & time filters.
-- ============================================================================

SELECT
  'QC4_Infection_Flags' AS check_name,
  COUNT(*) AS total_rows,

  SUM(COALESCE(sepsis3_flag, 0))   AS n_sepsis3,
  SUM(COALESCE(sirs_flag, 0))      AS n_sirs_ge2,
  SUM(COALESCE(suspicion_of_infection_flag, 0)) AS n_suspected_infection,

  ROUND(100.0 * SUM(COALESCE(sepsis3_flag, 0)) / COUNT(*), 1) AS pct_sepsis3,
  ROUND(100.0 * SUM(COALESCE(sirs_flag, 0)) / COUNT(*), 1)    AS pct_sirs_ge2,
  ROUND(100.0 * SUM(COALESCE(suspicion_of_infection_flag, 0)) / COUNT(*), 1)
    AS pct_suspected_infection
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`;



-- ============================================================================
-- QC5: SOFA Risk Category Distribution
-- Purpose:
--   • Check how patients are distributed across sofa_risk_category bins.
--   • Sanity check for category logic (Unknown / Low / Moderate / High / Very High).
-- ============================================================================

SELECT
  'QC5_SOFA_Risk_Categories' AS check_name,
  sofa_risk_category,
  COUNT(*) AS count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`
GROUP BY sofa_risk_category
ORDER BY count DESC;



-- ============================================================================
-- QC6: Severity vs 30-day Readmission (quick signal check)
-- Purpose:
--   • Not a rigorous analysis – just a sanity check:
--       Does higher severity roughly associate with higher readmission?
--   • we can later convert this into proper modelling or calibration.
-- ============================================================================

SELECT
  'QC6_SOFA_vs_Readmission' AS check_name,
  sofa_risk_category,
  COUNT(*) AS total_rows,
  SUM(readmit_30d_flag) AS readmit_count,
  ROUND(100.0 * SUM(readmit_30d_flag) / COUNT(*), 1) AS readmit_rate_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`
GROUP BY sofa_risk_category
ORDER BY total_rows DESC;


-- Charlson index buckets vs readmission
WITH charlson_binned AS (
  SELECT
    CASE
      WHEN charlson_comorbidity_index IS NULL THEN 'Missing'
      WHEN charlson_comorbidity_index = 0    THEN '0'
      WHEN charlson_comorbidity_index BETWEEN 1 AND 2 THEN '1–2'
      WHEN charlson_comorbidity_index BETWEEN 3 AND 4 THEN '3–4'
      ELSE '5+'
    END AS charlson_bucket,
    readmit_30d_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`
)

SELECT
  'QC6_Charlson_vs_Readmission' AS check_name,
  charlson_bucket,
  COUNT(*) AS total_rows,
  SUM(readmit_30d_flag) AS readmit_count,
  ROUND(100.0 * SUM(readmit_30d_flag) / COUNT(*), 1) AS readmit_rate_pct
FROM charlson_binned
GROUP BY charlson_bucket
ORDER BY
  CASE charlson_bucket
    WHEN 'Missing' THEN 0
    WHEN '0'       THEN 1
    WHEN '1–2'     THEN 2
    WHEN '3–4'     THEN 3
    ELSE 4
  END;




 -- for table 8 VIew creation prior column name auditing for creating table 8 :-


 -- Preview a single row (shows exact column names) from public physionet derived tables:
SELECT * FROM `physionet-data.mimiciv_3_1_derived.first_day_urine_output` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.urine_output_rate` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.first_day_vitalsign` LIMIT 1;
SELECT * FROM `physionet-data.mimiciv_3_1_derived.first_day_lab` LIMIT 1;

-- And from hosp / icu tables:
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions` LIMIT 3;
SELECT * FROM `physionet-data.mimiciv_3_1_icu.icustays` LIMIT 3;


-- 1) List columns for a single table we created (example: feature_labs_first_24h)
SELECT column_name, data_type, ordinal_position
FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'feature_labs_first_24h'
ORDER BY ordinal_position;

-- 2) Show columns for several of our feature tables at once
SELECT table_name, STRING_AGG(column_name, ', ' ORDER BY ordinal_position) AS columns
FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name IN (
  'feature_anthropometry',
  'feature_labs_first_24h',
  'feature_demographics_extended',
  'feature_medications_interventions',
  'feature_neurological'
)
GROUP BY table_name
ORDER BY table_name;

-- Quick existence check for candidate columns inside our dataset tables
WITH ask AS (
  SELECT 'feature_anthropometry' AS tbl, 'weight_kg' AS col UNION ALL
  SELECT 'feature_labs_first_24h', 'lactate_first_24h_max' UNION ALL
  SELECT 'feature_neurological', 'gcs_total_first_24h_min' UNION ALL
  SELECT 'feature_vitals_first_24h', 'meanbp_min' UNION ALL
  SELECT 'feature_medications_interventions', 'vasopressor_flag'
)
SELECT
  a.tbl AS table_name,
  a.col AS requested_column,
  CASE WHEN COUNT(c.column_name) > 0 THEN 1 ELSE 0 END AS exists_flag,
  STRING_AGG(c.column_name, ',') AS matched_columns -- will be null if not found
FROM ask a
LEFT JOIN `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS` c
  ON c.table_name = a.tbl AND LOWER(c.column_name) = LOWER(a.col)
GROUP BY a.tbl, a.col
ORDER BY a.tbl;


/* ============================================================================
   VIEW  : feature_prior_history_hemodynamics  (TABLE 8)
   PATH  : nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics

   PURPOSE
   -------
   This feature layer captures, for each index ICU stay:

     • Prior healthcare utilisation:
         - How often the patient has been admitted / in ICU in the last year.
         - How recently they were discharged before this index admission.
         - Whether this index stay is part of a “rapid re-use” pattern.

     • Early hemodynamics and perfusion:
         - Mean arterial pressure (MAP / MBP) in first 24h.
         - Lactate as a global hypoperfusion marker.
         - Urine output and weight-normalised urine output rate.

   Why this feature is important:
     • Frequent prior users with short gaps between discharges are classic
       high-readmission-risk phenotypes (“frequent flyers”).
     • Low MAP + high lactate + oliguria are hallmarks of shock and organ
       under-perfusion, which are strongly linked to post-ICU instability.
     • Combining utilisation history with early physiologic severity gives a
       very powerful signal for 30-day readmission risk.


   DATA SOURCES / INPUT TABLES
   ---------------------------
   COHORT DRIVER
     • nomadic-freedom-436306-g4.readmission30.feature_demographics_extended
       - One row per (subject_id, hadm_id, index_stay_id)
       - Provides: subject_id, hadm_id, index_stay_id, index_icu_intime,
                   admittime, readmit_30d_flag

   PRIOR HISTORY (LAST 12 MONTHS)
     • physionet-data.mimiciv_3_1_hosp.admissions
       - All prior/other hospital admissions per subject_id
       - Fields: subject_id, hadm_id, admittime, dischtime
     • physionet-data.mimiciv_3_1_icu.icustays
       - All ICU stays per subject_id
       - Fields: subject_id, hadm_id, stay_id, intime, outtime

   HEMODYNAMICS (FIRST 24h)
     • physionet-data.mimiciv_3_1_derived.first_day_vitalsign
       - stay-level aggregates over first 24h
       - We use: mbp_min, mbp_mean  (MAP)

   PERFUSION / RENAL FUNCTION
     • nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h
       - lactate_first_24h_max, elevated_lactate_flag
     • nomadic-freedom-436306-g4.readmission30.feature_anthropometry
       - weight_kg  (needed for mL/kg/hr)
     • physionet-data.mimiciv_3_1_derived.first_day_urine_output
       - stay-level urineoutput (total mL in first 24h)


   KEY DESIGN DECISIONS / ISSUES WE SOLVED
   ---------------------------------------
   1) 12-MONTH LOOKBACK WINDOWS
      • Prior hospital use can be very long-term. We chose 365 days as a
        practical window that balances:
          - capturing frequent flyers and chronic high users,
          - avoiding very old admissions that are less predictive.
      • Prior admissions are based on admittime < index admittime.
      • Prior ICU stays are based on intime < index_icu_intime.

   2) EXCLUDING THE CURRENT ADMISSION FROM PRIOR COUNTS
      • ICU stays table (icustays) also contains the current stay.
      • We explicitly exclude stays whose hadm_id = index hadm_id when
        counting prior ICU stays to avoid double-counting the index stay.

   3) “RECENT READMISSION” FLAGS (7D / 30D)
      • We distinguish between:
          - Prior *any* admission in last 12 months (chronic utilisation).
          - Prior admission in last 7 / 30 days (rapid bounce-back).
      • These are encoded as binary flags:
          - recent_readmission_flag_7d
          - recent_readmission_flag_30d

   4) USE OF DERIVED TABLES FOR MAP & URINE
      • We use derived.first_day_vitalsign and derived.first_day_urine_output
        rather than raw flowsheets to:
          - minimise joins and unit handling,
          - guarantee one row per stay_id with pre-aggregated 24h values.

   5) WEIGHT-NORMALISED URINE OUTPUT RATE
      • Oliguria is classically defined in mL/kg/hr.
      • We combine:
          - urineoutput (mL / 24h),
          - weight_kg from the anthropometry feature table,
        to compute urine_output_rate_ml_per_kg_hr and then oliguria_flag.

   6) SIMPLE “SHOCK” PHENOTYPE
      • We define shock_flag with a simple clinical rule:
          - mbp_first_24h_min < 65 AND lactate_first_24h_max > 2
      • This is not a sepsis definition, but a broad instability marker
        signalling high risk for deterioration and potential readmission.

   7) VIEW, NOT TABLE
      • Implemented as a VIEW to:
          - avoid storage / quota overhead,
          - keep logic central and auditable,
          - allow recomputation if upstream features change.


──────────────────────────────────────────────────────────────────────────────
OUTPUT COLUMNS (Short Definitions & Why They Matter)
──────────────────────────────────────────────────────────────────────────────

CORE KEYS
---------
• subject_id
    → Unique patient identifier.
      Why it matters: joins all feature layers across the pipeline.

• hadm_id
    → Hospital admission identifier for the index admission.
      Why it matters: modelling unit for 30-day readmission (per admission).

• index_stay_id
    → ICU stay identifier for the index ICU stay.
      Why it matters: links to all stay-level derived tables (vitals, urine).

• readmit_30d_flag
    → 1 if the patient was readmitted within 30 days of hospital discharge,
      else 0.
      Why it matters: this is your primary model target (label).


PRIOR HISTORY / UTILISATION
---------------------------
• prior_admissions_12m
    → Number of previous hospital admissions in the 365 days before the
      index admission.
      Why it matters:
        - High values capture “frequent flyer” behaviour and chronic
          complexity, which are strong predictors of readmission.

• prior_icu_stays_12m
    → Number of previous ICU stays in the 365 days before the index ICU stay
      (excluding the current admission).
      Why it matters:
        - Recurrent ICU use = fragile physiology / complex disease, often
          strongly associated with future unplanned readmissions.

• days_since_last_discharge
    → Days between the most recent prior hospital discharge and the index
      admission date (NULL if no prior stay).
      Why it matters:
        - Short intervals suggest bounce-back / unresolved issues.
        - Very long intervals may represent more isolated events.

• recent_readmission_flag_7d
    → 1 if any admission occurred in the 7 days before the index admission,
      else 0.
      Why it matters:
        - Captures very early bounce-backs and severe discharge failure.

• recent_readmission_flag_30d
    → 1 if any admission occurred in the 30 days before the index admission,
      else 0.
      Why it matters:
        - Identifies patients already in a high utilisation spiral.


ADMISSION FREQUENCY CATEGORY
----------------------------
• admission_frequency_category
    → Bucketed prior-admission phenotype:
        - 'Frequent_Flyer' : prior_admissions_12m ≥ 3
        - 'Occasional'     : 1–2 prior admissions
        - 'First_Time'     : 0 prior admissions
      Why it matters:
        - Easy-to-interpret feature for clinicians and for stratified
          readmission analysis.
        - Often directly linked to social determinants, complexity, and
          discharge planning needs.


HEMODYNAMIC MARKERS (MAP)
-------------------------
• mbp_first_24h_min
    → Minimum mean arterial pressure (MBP/MAP) in the first 24 hours of
      the index ICU stay (mmHg).
      Why it matters:
        - Low MAP is a direct marker of shock and haemodynamic instability.

• mbp_first_24h_mean
    → Mean MAP over the first 24 hours (mmHg).
      Why it matters:
        - Captures overall pressure support requirement and perfusion
          adequacy, rather than just a single nadir.


LACTATE (TISSUE PERFUSION)
--------------------------
• lactate_first_24h_max
    → Maximum lactate observed in the first 24 hours of the ICU stay
      (mmol/L).
      Why it matters:
        - Elevated lactate indicates hypoperfusion and cellular stress.
        - Persistent elevation is associated with poor outcomes and
          higher risk of post-ICU deterioration.

• elevated_lactate_flag
    → 1 if lactate_first_24h_max > 4 (threshold from labs feature table),
      else 0.
      Why it matters:
        - Simple binary severity signal, easy to interpret in dashboards
          and threshold-based risk rules.


URINE OUTPUT & RENAL PERFUSION
------------------------------
• urine_output_first_24h_ml
    → Total urine output in millilitres during the first 24 hours
      (stay-level, derived table).
      Why it matters:
        - Low output suggests poor renal perfusion / AKI, which is linked
          to both ICU mortality and post-discharge complications.

• urine_output_rate_ml_per_kg_hr
    → Weight-normalised urine output rate (mL/kg/hr) over first 24 hours:
         urine_output_first_24h_ml / (weight_kg * 24)
      Why it matters:
        - Standardises for body size.
        - Directly comparable with KDIGO oliguria thresholds used in
          critical care guidelines.

• oliguria_flag
    → 1 if urine_output_rate_ml_per_kg_hr < 0.5, else 0.
      Why it matters:
        - Classic oliguric AKI definition; strong prognostic marker for
          adverse outcomes and potential readmission.


COMBINED SHOCK MARKER
---------------------
• shock_flag
    → 1 if:
         mbp_first_24h_min < 65 AND lactate_first_24h_max > 2,
      else 0.
      Why it matters:
        - Encodes global haemodynamic compromise (low MAP + elevated
          lactate) as a single, interpretable binary flag.
        - Shock patients surviving to discharge often remain unstable
          and at high risk for early readmission.


AVAILABILITY FLAGS
------------------
• mbp_available_flag
    → 1 if mbp_first_24h_min is non-null, else 0.
      Why it matters:
        - Indicates whether MAP-based features can be trusted for a
          given stay; useful as a missingness indicator in models.

• lactate_available_flag
    → 1 if lactate_first_24h_max is non-null, else 0.
      Why it matters:
        - Lactate is not drawn for all patients; ordering itself is
          a signal of perceived severity.

• urine_output_available_flag
    → 1 if urine_output_first_24h_ml is non-null, else 0.
      Why it matters:
        - Missing urine output may indicate documentation or device
          issues; can be used in model as a data-quality / workflow signal.

• urine_rate_available_flag
    → 1 if both urine_output and weight_kg are present and >0, else 0.
      Why it matters:
        - Ensures that calculated mL/kg/hr is only used when both
          components are available and valid.


METADATA
--------
• feature_extraction_ts
    → Timestamp when this view was last evaluated (CURRENT_TIMESTAMP()).
      Why it matters:
        - Supports pipeline auditing and aligning model runs with feature
          generation time.


USAGE FOR READMISSION MODELING
------------------------------
• Direct numeric predictors:
    - prior_admissions_12m, prior_icu_stays_12m, days_since_last_discharge
    - mbp_first_24h_min, mbp_first_24h_mean
    - lactate_first_24h_max
    - urine_output_first_24h_ml, urine_output_rate_ml_per_kg_hr

• Categorical / binary features:
    - admission_frequency_category
    - recent_readmission_flag_7d / 30d
    - elevated_lactate_flag, oliguria_flag, shock_flag
    - availability flags (mbp / lactate / urine / urine_rate)

• Typical modelling roles:
    - Capture chronic system utilisation (frequent flyers).
    - Encode early multi-organ dysfunction (shock, AKI).
    - Provide high-level phenotypes (e.g., “shock + oliguria + short
      days_since_last_discharge”) that are highly interpretable to
      clinicians and business stakeholders when explaining model outputs.
============================================================================ */


CREATE OR REPLACE VIEW
  `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics` AS

/* ---------------------------------------------------------------------------
   0) COHORT: pull canonical row set from feature_demographics_extended
   --------------------------------------------------------------------------- */
WITH cohort AS (
  SELECT
    subject_id,
    hadm_id,
    index_stay_id,
    index_icu_intime,
    admittime,
    readmit_30d_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`
),

/* ---------------------------------------------------------------------------
   1) PRIOR ADMISSION & ICU HISTORY (12-month lookback)
   --------------------------------------------------------------------------- */
prior_history AS (
  SELECT
    c.subject_id,
    c.hadm_id,

    -- Distinct PRIOR admissions within 12 months before index admittime
    COUNT(DISTINCT IF(
      a.admittime < c.admittime
      AND a.admittime >= TIMESTAMP_SUB(c.admittime, INTERVAL 365 DAY),
      a.hadm_id,
      NULL
    )) AS prior_admissions_12m,

    -- Distinct PRIOR ICU stays within 12 months before index ICU intime
    COUNT(DISTINCT IF(
      i.intime < c.index_icu_intime
      AND i.intime >= TIMESTAMP_SUB(c.index_icu_intime, INTERVAL 365 DAY)
      AND COALESCE(i.hadm_id, -1) != COALESCE(c.hadm_id, -1),  -- exclude current admission
      i.stay_id,
      NULL
    )) AS prior_icu_stays_12m,

    -- Days since last discharge (NULL if no previous discharge)
    DATE_DIFF(
      DATE(c.admittime),
      DATE(
        MAX(
          IF(a.admittime < c.admittime, a.dischtime, NULL)
        )
      ),
      DAY
    ) AS days_since_last_discharge,

    -- Recent readmission flags (7-day and 30-day windows before index)
    MAX(
      IF(
        a.admittime >= TIMESTAMP_SUB(c.admittime, INTERVAL 7 DAY)
        AND a.admittime < c.admittime,
        1, 0
      )
    ) AS recent_readmission_flag_7d,

    MAX(
      IF(
        a.admittime >= TIMESTAMP_SUB(c.admittime, INTERVAL 30 DAY)
        AND a.admittime < c.admittime,
        1, 0
      )
    ) AS recent_readmission_flag_30d

  FROM cohort c
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON c.subject_id = a.subject_id
  LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
    ON c.subject_id = i.subject_id
  GROUP BY
    c.subject_id,
    c.hadm_id,
    c.admittime,
    c.index_icu_intime
),

/* ---------------------------------------------------------------------------
   2) FIRST-DAY VITALS: MAP (mbp_min, mbp_mean)
   --------------------------------------------------------------------------- */
vitals_first_day AS (
  SELECT
    stay_id,
    subject_id,
    SAFE_CAST(mbp_min  AS FLOAT64) AS mbp_first_24h_min,
    SAFE_CAST(mbp_mean AS FLOAT64) AS mbp_first_24h_mean
  FROM `physionet-data.mimiciv_3_1_derived.first_day_vitalsign`
  WHERE stay_id IS NOT NULL
),

/* ---------------------------------------------------------------------------
   3) FIRST-DAY LACTATE: from your labs feature table
   --------------------------------------------------------------------------- */
lactate_first_day AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(lactate_first_24h_max AS FLOAT64) AS lactate_first_24h_max,
    SAFE_CAST(elevated_lactate_flag AS INT64)   AS elevated_lactate_flag
  FROM `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`
),

/* ---------------------------------------------------------------------------
   4) ANTHROPOMETRY: weight (needed for mL/kg/hr)
   --------------------------------------------------------------------------- */
anthro_small AS (
  SELECT
    subject_id,
    hadm_id,
    SAFE_CAST(weight_kg AS FLOAT64) AS weight_kg
  FROM `nomadic-freedom-436306-g4.readmission30.feature_anthropometry`
),

/* ---------------------------------------------------------------------------
   5) FIRST-DAY URINE OUTPUT (total mL in first 24h)
      NOTE: uses "urineoutput" column from first_day_urine_output.
   --------------------------------------------------------------------------- */
urine_first_day AS (
  SELECT
    subject_id,
    stay_id,
    SAFE_CAST(urineoutput AS FLOAT64) AS urine_output_first_24h_ml
  FROM `physionet-data.mimiciv_3_1_derived.first_day_urine_output`
)

-- ---------------------------------------------------------------------------
-- 6) FINAL ASSEMBLY
-- ---------------------------------------------------------------------------
SELECT
  c.subject_id,
  c.hadm_id,
  c.index_stay_id,
  c.readmit_30d_flag,

  -- PRIOR ADMISSION HISTORY
  ph.prior_admissions_12m,
  ph.prior_icu_stays_12m,
  ph.days_since_last_discharge,
  ph.recent_readmission_flag_7d,
  ph.recent_readmission_flag_30d,

  -- Admission frequency bucket (simple prior-use phenotype)
  CASE
    WHEN ph.prior_admissions_12m >= 3 THEN 'Frequent_Flyer'
    WHEN ph.prior_admissions_12m >= 1 THEN 'Occasional'
    ELSE 'First_Time'
  END AS admission_frequency_category,

  -- HEMODYNAMIC MARKERS
  v.mbp_first_24h_min,
  v.mbp_first_24h_mean,

  -- LACTATE
  l.lactate_first_24h_max,
  l.elevated_lactate_flag,

  -- URINE OUTPUT (absolute)
  u.urine_output_first_24h_ml,

  -- URINE OUTPUT RATE (mL/kg/hr)
  CASE
    WHEN u.urine_output_first_24h_ml IS NOT NULL
     AND a.weight_kg IS NOT NULL
     AND a.weight_kg > 0
    THEN u.urine_output_first_24h_ml / (a.weight_kg * 24.0)
    ELSE NULL
  END AS urine_output_rate_ml_per_kg_hr,

  -- Oliguria: < 0.5 mL/kg/hr
  CASE
    WHEN u.urine_output_first_24h_ml IS NOT NULL
     AND a.weight_kg IS NOT NULL
     AND a.weight_kg > 0
     AND (u.urine_output_first_24h_ml / (a.weight_kg * 24.0)) < 0.5
    THEN 1
    ELSE 0
  END AS oliguria_flag,

  -- Simple "shock" flag: MAP < 65 AND lactate > 2 in first 24h
  CASE
    WHEN v.mbp_first_24h_min IS NOT NULL
     AND l.lactate_first_24h_max IS NOT NULL
     AND v.mbp_first_24h_min < 65
     AND l.lactate_first_24h_max > 2
    THEN 1
    ELSE 0
  END AS shock_flag,

  -- Availability flags
  CASE WHEN v.mbp_first_24h_min           IS NOT NULL THEN 1 ELSE 0 END AS mbp_available_flag,
  CASE WHEN l.lactate_first_24h_max       IS NOT NULL THEN 1 ELSE 0 END AS lactate_available_flag,
  CASE WHEN u.urine_output_first_24h_ml   IS NOT NULL THEN 1 ELSE 0 END AS urine_output_available_flag,
  CASE
    WHEN u.urine_output_first_24h_ml IS NOT NULL
     AND a.weight_kg IS NOT NULL
     AND a.weight_kg > 0
    THEN 1
    ELSE 0
  END AS urine_rate_available_flag,

  CURRENT_TIMESTAMP() AS feature_extraction_ts

FROM cohort c
LEFT JOIN prior_history ph
  ON c.subject_id = ph.subject_id
 AND c.hadm_id   = ph.hadm_id
LEFT JOIN vitals_first_day v
  ON c.index_stay_id = v.stay_id
LEFT JOIN lactate_first_day l
  ON c.subject_id = l.subject_id
 AND c.hadm_id   = l.hadm_id
LEFT JOIN anthro_small a
  ON c.subject_id = a.subject_id
 AND c.hadm_id   = a.hadm_id
LEFT JOIN urine_first_day u
  ON c.subject_id   = u.subject_id
 AND c.index_stay_id = u.stay_id;




-- =====================================================================
-- QUICK AUDIT: peek at a few rows
-- =====================================================================
SELECT * 
FROM `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics`
LIMIT 100;

-- =====================================================================
-- QC 1: Overall coverage for prior history + hemodynamics + urine
-- =====================================================================
SELECT
  'QC1_Coverage' AS check_name,
  COUNT(*) AS total_rows,

  -- prior history
  COUNTIF(prior_admissions_12m IS NOT NULL) AS prior_admissions_nonnull,
  COUNTIF(prior_icu_stays_12m IS NOT NULL)  AS prior_icu_nonnull,
  COUNTIF(days_since_last_discharge IS NOT NULL) AS days_since_last_discharge_nonnull,

  -- availability flags
  COUNTIF(mbp_available_flag = 1)         AS mbp_available_count,
  COUNTIF(lactate_available_flag = 1)     AS lactate_available_count,
  COUNTIF(urine_output_available_flag = 1) AS urine_output_available_count,
  COUNTIF(urine_rate_available_flag = 1)   AS urine_rate_available_count,

  -- basic prevalence
  SUM(oliguria_flag) AS oliguria_count,
  SUM(shock_flag)    AS shock_count
FROM `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics`;

-- =====================================================================
-- QC 2: Admission frequency buckets & 30-day readmission rate
-- =====================================================================
SELECT
  'QC2_Admission_Frequency' AS check_name,
  admission_frequency_category,
  COUNT(*) AS n,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
  ROUND(100.0 * SUM(readmit_30d_flag) / NULLIF(COUNT(*),0), 2) AS readmit_rate_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics`
GROUP BY admission_frequency_category
ORDER BY n DESC;

-- =====================================================================
-- QC 3: Shock prevalence & readmission in the shock cohort
-- =====================================================================
SELECT
  'QC3_Shock' AS check_name,
  COUNT(*) AS total_rows,
  SUM(shock_flag) AS shock_count,
  ROUND(100.0 * SUM(shock_flag) / COUNT(*), 2) AS shock_pct,

  -- readmissions among shock patients
  SUM(CASE WHEN shock_flag = 1 AND readmit_30d_flag = 1 THEN 1 ELSE 0 END) AS shock_readmit_count,
  ROUND(
    100.0 * SUM(CASE WHEN shock_flag = 1 AND readmit_30d_flag = 1 THEN 1 ELSE 0 END)
    / NULLIF(SUM(CASE WHEN shock_flag = 1 THEN 1 ELSE 0 END), 0),
    2
  ) AS shock_readmit_rate_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics`;

-- =====================================================================
-- QC 4: Oliguria prevalence & readmission
-- =====================================================================
SELECT
  'QC4_Oliguria' AS check_name,
  COUNT(*) AS total_rows,
  SUM(oliguria_flag) AS oliguria_count,
  ROUND(100.0 * SUM(oliguria_flag) / COUNT(*), 2) AS oliguria_pct,

  SUM(CASE WHEN oliguria_flag = 1 AND readmit_30d_flag = 1 THEN 1 ELSE 0 END) AS oliguria_readmit_count,
  ROUND(
    100.0 * SUM(CASE WHEN oliguria_flag = 1 AND readmit_30d_flag = 1 THEN 1 ELSE 0 END)
    / NULLIF(SUM(CASE WHEN oliguria_flag = 1 THEN 1 ELSE 0 END), 0),
    2
  ) AS oliguria_readmit_rate_pct
FROM `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics`;

-- =====================================================================
-- QC 5: Basic distributions for continuous vars (MAP, lactate, urine)
--      (just to get a feel for ranges; you can plot later in Python)
-- =====================================================================
SELECT
  'QC5_Distributions' AS check_name,
  COUNT(*) AS total_rows,
  ROUND(AVG(mbp_first_24h_min), 1)   AS avg_map_min,
  ROUND(AVG(mbp_first_24h_mean), 1)  AS avg_map_mean,
  ROUND(AVG(lactate_first_24h_max), 2) AS avg_lactate_max,
  ROUND(AVG(urine_output_first_24h_ml), 1) AS avg_urine_ml,
  ROUND(AVG(urine_output_rate_ml_per_kg_hr), 3) AS avg_urine_rate_ml_kg_hr
FROM `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics`;



-- Treat all the duplicated Col feature in final table 
/* ============================================================================
   DIAGNOSTIC: Find duplicate column names across feature tables/views
   Dataset : nomadic-freedom-436306-g4.readmission30
   Goal    : See which column_name appears in >1 feature table/view
   ============================================================================ */

WITH feature_tables AS (
  SELECT 'feature_demographics_extended'         AS table_name UNION ALL
  SELECT 'feature_anthropometry'                UNION ALL
  SELECT 'feature_vitals_first_24h'             UNION ALL
  SELECT 'feature_labs_first_24h'               UNION ALL
  SELECT 'feature_neurological'                 UNION ALL
  SELECT 'feature_medications_interventions'    UNION ALL
  SELECT 'feature_comorbidities_severity'       UNION ALL
  SELECT 'feature_prior_history_hemodynamics'
),

all_columns AS (
  SELECT
    c.table_name,
    c.column_name
  FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS` AS c
  JOIN feature_tables ft
    ON c.table_name = ft.table_name
)

SELECT
  column_name,
  COUNT(DISTINCT table_name) AS n_tables,
  STRING_AGG(table_name, ', ' ORDER BY table_name) AS tables_with_column
FROM all_columns
GROUP BY column_name
HAVING COUNT(DISTINCT table_name) > 1
ORDER BY n_tables DESC, column_name;



/* =============================================================================
   TABLE : model_dataset_readmission_30d
   PATH  : nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d

   PURPOSE
   -------
   Final, wide, ML-ready feature table for **30-day ICU readmission**.

   This version:
     • Removes pipeline-only metadata:
         - cohort_creation_timestamp
         - model_feature_extraction_ts (dropped entirely)
     • Ensures the **target columns** are the last two fields in the schema:
         - readmit_30d_flag
         - days_to_30d_readmission
     • Keeps each feature family grouped together by layer so analysts can
       quickly scan related variables (e.g. all urine features together, all
       GCS features together, etc.).

   KEYS & GRAIN
   ------------
   • 1 row per index ICU stay:
       (subject_id, hadm_id, index_stay_id)
   • Driver: feature_demographics_extended (d)

   FEATURE LAYERS (column groups stay logically clustered)
   -------------------------------------------------------
   d = feature_demographics_extended      → keys, demographics, LOS, target
   a = feature_anthropometry              → height/weight/BMI + flags
   v = feature_vitals_first_24h           → HR/BP/MAP/RR/SpO2/FiO2/vent + glucose_vitals
   l = feature_labs_first_24h             → labs, canonical lactate & lab glucose
   n = feature_neurological               → GCS totals/components/flags
   m = feature_medications_interventions  → meds/vaso/organ support/intensity
   cs = feature_comorbidities_severity    → Charlson, mapping flags, severity, sepsis
   h = feature_prior_history_hemodynamics → prior admissions, urine output, oliguria, shock

   DUPLICATE / METADATA HANDLING
   ------------------------------
   • Keys & target:
       - Keep only from d (demographics).
       - Remove from all other layers via EXCEPT(...).
   • Lactate:
       - Canonical: labs table (l).
       - Drop lactate_* from h in the final SELECT.
   • MAP:
       - Canonical: vitals table (v) → mbp_first_24h_mean.
       - Drop mbp_first_24h_mean from h in the final SELECT.
   • Glucose:
       - Labs (l) is canonical for lab-based glucose.
       - Vitals (v) glucose kept but renamed with *_vitals suffix.
   • Timestamps:
       - Drop all per-layer feature_extraction_timestamp / feature_extraction_ts.
       - Do **not** add a global model_feature_extraction_ts (per your request).
   • Target ordering:
       - Exclude readmit_30d_flag and days_to_30d_readmission from d.*
       - Re-append them at the very end so they are the **last two columns**.

   RESULT
   ------
   • Clean, non-duplicated schema:
       - Keys at the top
       - Feature blocks grouped by feature layer
       - Targets at the very end:
           [... many features ...], readmit_30d_flag, days_to_30d_readmission
   • Ready to EXPORT to Parquet/CSV and load directly into Python for modeling.
   =============================================================================
*/

CREATE OR REPLACE TABLE
  `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d` AS

SELECT
  /* ----------------------------------------------------------------------
     1) DEMOGRAPHICS (DRIVER)
        - Keep keys & all core demographic / LOS fields
        - Drop:
            • feature_extraction_timestamp
            • cohort_creation_timestamp
            • readmit_30d_flag
            • days_to_30d_readmission
          so we can place the last two at the bottom.
     ---------------------------------------------------------------------- */
  d.* EXCEPT(
    feature_extraction_timestamp,
    cohort_creation_timestamp,
    readmit_30d_flag,
    days_to_30d_readmission
  ),

  /* ----------------------------------------------------------------------
     2) ANTHROPOMETRY – height/weight/BMI grouped together
        - Drop duplicate keys + local timestamp.
     ---------------------------------------------------------------------- */
  a.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    feature_extraction_timestamp
  ),

  /* ----------------------------------------------------------------------
     3) VITALS FIRST 24H – all vitals + vent + glucose (vitals)
        - Drop keys, timestamp, and raw glucose fields (we’ll re-add them
          with *_vitals suffix for clarity).
     ---------------------------------------------------------------------- */
  v.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    feature_extraction_timestamp,
    glucose_first_24h_min,
    glucose_first_24h_max,
    glucose_first_24h_mean,
    hypoglycemia_flag,
    hyperglycemia_flag
  ),

  -- Vitals-based glucose re-added, clearly grouped and suffixed
  v.glucose_first_24h_min  AS glucose_first_24h_min_vitals,
  v.glucose_first_24h_max  AS glucose_first_24h_max_vitals,
  v.glucose_first_24h_mean AS glucose_first_24h_mean_vitals,
  v.hypoglycemia_flag      AS hypoglycemia_flag_vitals,
  v.hyperglycemia_flag     AS hyperglycemia_flag_vitals,

  /* ----------------------------------------------------------------------
     4) LABS FIRST 24H – hematology, electrolytes, renal, liver, coag, lactate
        - Canonical owner of lactate_* and lab glucose_*.
        - Drop keys, readmit_30d_flag, local feature_extraction_ts.
     ---------------------------------------------------------------------- */
  l.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    readmit_30d_flag,
    feature_extraction_ts
  ),

  /* ----------------------------------------------------------------------
     5) NEUROLOGICAL (GCS) – all GCS columns stay next to each other
        - Drop keys, readmit_30d_flag, timestamp.
     ---------------------------------------------------------------------- */
  n.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    readmit_30d_flag,
    feature_extraction_timestamp
  ),

  /* ----------------------------------------------------------------------
     6) MEDICATIONS & INTERVENTIONS – ACEI/ARB/ATB, vaso, organ support
        - Drop keys, readmit_30d_flag, timestamp.
     ---------------------------------------------------------------------- */
  m.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    readmit_30d_flag,
    feature_extraction_timestamp
  ),

  /* ----------------------------------------------------------------------
     7) COMORBIDITIES & SEVERITY – Charlson, ICD flags, SOFA, APSIII, etc.
        - Drop keys, readmit_30d_flag, local feature_extraction_ts.
     ---------------------------------------------------------------------- */
  cs.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    readmit_30d_flag,
    feature_extraction_ts
  ),

  /* ----------------------------------------------------------------------
     8) PRIOR HISTORY & HEMODYNAMICS – prior admits, urine, oliguria, shock
        - Keep prior_* and urine_* features grouped together.
        - Drop:
            • keys
            • readmit_30d_flag
            • feature_extraction_ts
            • lactate_first_24h_max            (labs is canonical)
            • lactate_available_flag           (labs is canonical)
            • elevated_lactate_flag            (labs is canonical)
            • mbp_first_24h_mean               (vitals is canonical)
     ---------------------------------------------------------------------- */
  h.* EXCEPT(
    subject_id,
    hadm_id,
    index_stay_id,
    readmit_30d_flag,
    feature_extraction_ts,
    lactate_first_24h_max,
    lactate_available_flag,
    elevated_lactate_flag,
    mbp_first_24h_mean
  ),

  /* ----------------------------------------------------------------------
     9) TARGET COLUMNS – placed LAST in the schema
        - readmit_30d_flag         → binary target
        - days_to_30d_readmission  → continuous time-to-event style feature
     ---------------------------------------------------------------------- */
  d.readmit_30d_flag,
  d.days_to_30d_readmission

FROM `nomadic-freedom-436306-g4.readmission30.feature_demographics_extended`           AS d
LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_anthropometry`              AS a
  ON d.subject_id    = a.subject_id
 AND d.hadm_id       = a.hadm_id
 AND d.index_stay_id = a.index_stay_id

LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_vitals_first_24h`           AS v
  ON d.subject_id    = v.subject_id
 AND d.hadm_id       = v.hadm_id
 AND d.index_stay_id = v.index_stay_id

LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_labs_first_24h`             AS l
  ON d.subject_id    = l.subject_id
 AND d.hadm_id       = l.hadm_id
 AND d.index_stay_id = l.index_stay_id

LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_neurological`               AS n
  ON d.subject_id    = n.subject_id
 AND d.hadm_id       = n.hadm_id
 AND d.index_stay_id = n.index_stay_id

LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_medications_interventions`  AS m
  ON d.subject_id    = m.subject_id
 AND d.hadm_id       = m.hadm_id
 AND d.index_stay_id = m.index_stay_id

LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_comorbidities_severity`     AS cs
  ON d.subject_id    = cs.subject_id
 AND d.hadm_id       = cs.hadm_id
 AND d.index_stay_id = cs.index_stay_id

LEFT JOIN `nomadic-freedom-436306-g4.readmission30.feature_prior_history_hemodynamics` AS h
  ON d.subject_id    = h.subject_id
 AND d.hadm_id       = h.hadm_id
 AND d.index_stay_id = h.index_stay_id;


-- Quality Checks ( FINAL )

-- =============================================================================
-- QC1: Row Count & Key Uniqueness
--  - Confirms:
--      • row count matches cohort
--      • (subject_id, hadm_id, index_stay_id) is unique
-- =============================================================================

-- 1A) Compare row counts to cohort
SELECT
  'QC1_RowCount' AS check_name,
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`) AS cohort_rows,
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`) AS model_rows,
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`)
  -
  (SELECT COUNT(*) FROM `nomadic-freedom-436306-g4.readmission30.mimiciv_index_cohort_30d`) AS row_count_diff;

-- 1B) Check for duplicate keys (should return 0 rows)
SELECT
  'QC1_KeyUniqueness' AS check_name,
  subject_id,
  hadm_id,
  index_stay_id,
  COUNT(*) AS cnt
FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`
GROUP BY subject_id, hadm_id, index_stay_id
HAVING cnt > 1
LIMIT 50;


-- =============================================================================
-- QC2: Target Distribution (baseline prevalence)
--  - How many readmissions vs non-readmissions?
--  - What is the overall 30-day readmission rate?
--  - Are there any NULL targets?
-- =============================================================================
SELECT
  'QC2_Target_Distribution' AS check_name,
  COUNT(*) AS total_rows,
  COUNTIF(readmit_30d_flag = 1) AS n_readmit,
  COUNTIF(readmit_30d_flag = 0) AS n_not_readmit,
  COUNTIF(readmit_30d_flag IS NULL) AS n_target_null,
  ROUND(100.0 * COUNTIF(readmit_30d_flag = 1) / NULLIF(COUNT(*),0), 2) AS readmit_rate_pct,
  ROUND(AVG(CAST(days_to_30d_readmission AS FLOAT64)), 2) AS avg_days_to_30d_readmission
FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`;


-- =============================================================================
-- QC3: Core Feature Coverage by Domain
--  - Quick completeness check for **representative fields** from each layer:
--      • Anthropometry
--      • Vitals
--      • Labs
--      • GCS
--      • Meds / Interventions
--      • Comorbidities / Severity
--      • Prior history / Urine / Shock
-- =============================================================================
SELECT
  'QC3_Core_Feature_Coverage' AS check_name,
  COUNT(*) AS total_rows,

  -- Anthropometry
  COUNTIF(height_cm IS NOT NULL)                  AS n_height_nonnull,
  COUNTIF(weight_kg IS NOT NULL)                  AS n_weight_nonnull,
  COUNTIF(bmi IS NOT NULL)                        AS n_bmi_nonnull,

  -- Vitals
  COUNTIF(hr_first_24h_mean IS NOT NULL)          AS n_hr_nonnull,
  COUNTIF(mbp_first_24h_mean IS NOT NULL)         AS n_map_nonnull,
  COUNTIF(spo2_first_24h_mean IS NOT NULL)        AS n_spo2_nonnull,

  -- Labs
  COUNTIF(creatinine_first_24h_max IS NOT NULL)   AS n_creatinine_nonnull,
  COUNTIF(lactate_first_24h_max IS NOT NULL)      AS n_lactate_nonnull,
  COUNTIF(glucose_first_24h_max IS NOT NULL)      AS n_glucose_lab_nonnull,

  -- Neurological (GCS)
  COUNTIF(gcs_total_first_24h IS NOT NULL)        AS n_gcs_total_nonnull,

  -- Medications / Interventions
  COUNTIF(vasopressor_24h_flag = 1)               AS n_vasopressor_24h,
  COUNTIF(mechanical_ventilation_24h_flag = 1)    AS n_mech_vent_24h,

  -- Comorbidities & severity
  COUNTIF(charlson_comorbidity_index IS NOT NULL) AS n_charlson_nonnull,
  COUNTIF(sofa_score_first_24h IS NOT NULL)       AS n_sofa_nonnull,

  -- Prior history / urine / shock
  COUNTIF(prior_admissions_12m IS NOT NULL)       AS n_prior_adm_nonnull,
  COUNTIF(urine_output_first_24h_ml IS NOT NULL)  AS n_urine_ml_nonnull,
  COUNTIF(urine_output_rate_ml_per_kg_hr IS NOT NULL) AS n_urine_rate_nonnull,
  COUNTIF(shock_flag = 1)                         AS n_shock_flag

FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`;


-- =============================================================================
-- QC4: Basic Clinical Range Sanity for Key Continuous Variables
--  - Sanity-band checks for a few core signals:
--      HR, creatinine, lactate, urine rate, SOFA
--  - This is not a hard filter, just a smoke test for extraction bugs.
-- =============================================================================
SELECT
  'QC4_Clinical_Ranges' AS check_name,
  COUNT(*) AS total_rows,

  -- Heart rate (beats/min) - very loose adult bounds
  COUNTIF(hr_first_24h_mean < 30 OR hr_first_24h_mean > 250) AS hr_out_of_range,

  -- Creatinine (mg/dL) - extreme tails only
  COUNTIF(creatinine_first_24h_max < 0.1 OR creatinine_first_24h_max > 20) AS creatinine_out_of_range,

  -- Lactate (mmol/L)
  COUNTIF(lactate_first_24h_max < 0 OR lactate_first_24h_max > 40) AS lactate_out_of_range,

  -- Urine output rate (mL/kg/hr)
  COUNTIF(urine_output_rate_ml_per_kg_hr < 0 OR urine_output_rate_ml_per_kg_hr > 20) AS urine_rate_out_of_range,

  -- SOFA (0–24 typical)
  COUNTIF(sofa_score_first_24h IS NOT NULL AND (sofa_score_first_24h < 0 OR sofa_score_first_24h > 30)) AS sofa_out_of_range
FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`;


-- =============================================================================
-- QC5: Data Availability vs Target (is missingness biased by outcome?)
--  - Check if key features are much more/less available in readmitted vs not.
--  - Can reveal subtle data leakage / documentation bias.
-- =============================================================================
SELECT
  'QC5_Availability_vs_Target' AS check_name,
  readmit_30d_flag,

  COUNT(*) AS n_rows,

  -- Coverage by outcome
  ROUND(100.0 * COUNTIF(height_cm IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_height_nonnull,
  ROUND(100.0 * COUNTIF(mbp_first_24h_mean IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_map_nonnull,
  ROUND(100.0 * COUNTIF(creatinine_first_24h_max IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_creatinine_nonnull,
  ROUND(100.0 * COUNTIF(lactate_first_24h_max IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_lactate_nonnull,
  ROUND(100.0 * COUNTIF(gcs_total_first_24h IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_gcs_nonnull,
  ROUND(100.0 * COUNTIF(sofa_score_first_24h IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_sofa_nonnull,
  ROUND(100.0 * COUNTIF(urine_output_rate_ml_per_kg_hr IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS pct_urine_rate_nonnull
FROM `nomadic-freedom-436306-g4.readmission30.model_dataset_readmission_30d`
GROUP BY readmit_30d_flag
ORDER BY readmit_30d_flag DESC;


-- =============================================================================
-- QC6: Schema Snapshot (column count & listing)
--  - Simple way to:
--      • Confirm final schema size (wide feature count)
--      • Inspect columns & ordering before export to Python/Parquet.
-- =============================================================================

-- 6A) Column count
SELECT
  'QC6_Schema_ColumnCount' AS check_name,
  COUNT(*) AS n_columns
FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'model_dataset_readmission_30d';

-- 6B) Column listing in final order
SELECT
  'QC6_Schema_Columns' AS check_name,
  ordinal_position,
  column_name,
  data_type
FROM `nomadic-freedom-436306-g4.readmission30.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'model_dataset_readmission_30d'
ORDER BY ordinal_position;







