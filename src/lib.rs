use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single cloud-resolution correction record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Correction {
    pub id: uuid::Uuid,
    pub room_id: String,
    pub tile_id: String,
    pub original_label: String,
    pub expected_label: String,
    pub cloud_response: String,
    pub confidence: f64,
    pub timestamp: u64,
}

impl Correction {
    pub fn new(
        room_id: impl Into<String>,
        tile_id: impl Into<String>,
        original_label: impl Into<String>,
        expected_label: impl Into<String>,
        cloud_response: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            room_id: room_id.into(),
            tile_id: tile_id.into(),
            original_label: original_label.into(),
            expected_label: expected_label.into(),
            cloud_response: cloud_response.into(),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// A training example extracted from corrections for LoRA adapter training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingExample {
    pub input: String,
    pub expected_output: String,
    pub confidence: f64,
    pub tile_id: String,
}

/// Configuration for when distillation should be triggered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistillationConfig {
    /// Minimum number of corrections required before distillation can run.
    pub min_corrections: usize,
    /// Optional: maximum time since last distillation before triggering (seconds).
    pub max_interval_secs: Option<u64>,
    /// Optional: minimum correction diversity (unique tile_ids) required.
    pub min_diversity: Option<usize>,
}

impl DistillationConfig {
    pub fn new(min_corrections: usize) -> Self {
        Self {
            min_corrections,
            max_interval_secs: None,
            min_diversity: None,
        }
    }

    pub fn with_interval(mut self, secs: u64) -> Self {
        self.max_interval_secs = Some(secs);
        self
    }

    pub fn with_min_diversity(mut self, n: usize) -> Self {
        self.min_diversity = Some(n);
        self
    }

    /// Validate config: min_corrections must be > 0, interval must be > 0 if set, diversity must be > 0 if set.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_corrections == 0 {
            return Err("min_corrections must be > 0".into());
        }
        if let Some(i) = self.max_interval_secs {
            if i == 0 {
                return Err("max_interval_secs must be > 0".into());
            }
        }
        if let Some(d) = self.min_diversity {
            if d == 0 {
                return Err("min_diversity must be > 0".into());
            }
        }
        Ok(())
    }
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            min_corrections: 50,
            max_interval_secs: Some(3600),
            min_diversity: Some(5),
        }
    }
}

/// Metadata about a completed distillation round.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistillationRecord {
    pub id: uuid::Uuid,
    pub room_id: String,
    pub corrections_consumed: usize,
    pub accuracy_before: f64,
    pub accuracy_after: f64,
    pub adapter_size_bytes: u64,
    pub timestamp: u64,
    pub round_number: usize,
}

/// Running statistics for the distillation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DistillationStats {
    pub total_corrections: usize,
    pub distillations_performed: usize,
    pub accuracy_trajectory: Vec<f64>,
}

impl DistillationStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// What percentage of readings are now resolved locally (latest accuracy).
    pub fn autonomy_percentage(&self) -> f64 {
        self.accuracy_trajectory
            .last()
            .copied()
            .unwrap_or(0.0)
            * 100.0
    }

    pub fn latest_accuracy(&self) -> f64 {
        self.accuracy_trajectory.last().copied().unwrap_or(0.0)
    }
}

/// Append-only log of corrections per room.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorrectionStore {
    corrections: Vec<Correction>,
    room_id: String,
    last_distillation_timestamp: Option<u64>,
}

impl CorrectionStore {
    pub fn new(room_id: impl Into<String>) -> Self {
        Self {
            corrections: Vec::new(),
            room_id: room_id.into(),
            last_distillation_timestamp: None,
        }
    }

    /// Log a new cloud correction.
    pub fn add(&mut self, correction: Correction) {
        self.corrections.push(correction);
    }

    /// Number of corrections stored.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// Get all corrections.
    pub fn corrections(&self) -> &[Correction] {
        &self.corrections
    }

    /// Number of unique tile_ids in the store (diversity measure).
    pub fn diversity(&self) -> usize {
        let unique: std::collections::HashSet<_> =
            self.corrections.iter().map(|c| &c.tile_id).collect();
        unique.len()
    }

    /// Average confidence of stored corrections.
    pub fn avg_confidence(&self) -> f64 {
        if self.corrections.is_empty() {
            return 0.0;
        }
        self.corrections.iter().map(|c| c.confidence).sum::<f64>() / self.corrections.len() as f64
    }

    /// Check if distillation should be triggered based on config thresholds.
    pub fn should_distill(&self, config: &DistillationConfig) -> bool {
        if self.corrections.len() < config.min_corrections {
            return false;
        }

        // Check diversity threshold
        if let Some(min_div) = config.min_diversity {
            if self.diversity() < min_div {
                return false;
            }
        }

        // Check time-based interval
        if let Some(interval) = config.max_interval_secs {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let last = self.last_distillation_timestamp.unwrap_or(0);
            if now - last < interval {
                return false;
            }
        }

        true
    }

    /// Format corrections for LoRA training.
    pub fn export_training_data(&self) -> Vec<TrainingExample> {
        self.corrections
            .iter()
            .map(|c| TrainingExample {
                input: format!("tile:{}:{}", c.room_id, c.tile_id),
                expected_output: c.expected_label.clone(),
                confidence: c.confidence,
                tile_id: c.tile_id.clone(),
            })
            .collect()
    }

    /// Mark that a distillation round has occurred (updates timestamp).
    pub fn mark_distilled(&mut self) {
        self.last_distillation_timestamp = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }

    /// Group corrections by tile_id.
    pub fn by_tile(&self) -> HashMap<String, Vec<&Correction>> {
        let mut map: HashMap<String, Vec<&Correction>> = HashMap::new();
        for c in &self.corrections {
            map.entry(c.tile_id.clone()).or_default().push(c);
        }
        map
    }
}

/// Simulate accuracy improvement over multiple distillation rounds.
///
/// Starts at `base_accuracy` (e.g., 0.60). Each round improves based on:
/// - Number of corrections available
/// - Diversity of corrections
/// - Diminishing returns (logarithmic curve)
pub fn simulate_distillation(
    corrections: usize,
    diversity: usize,
    rounds: usize,
    base_accuracy: f64,
) -> DistillationStats {
    let mut stats = DistillationStats::new();
    stats.total_corrections = corrections;

    let target = 0.99;

    let mut current = base_accuracy;
    for round in 1..=rounds {
        let diversity_factor = (diversity as f64 + 1.0).ln() / (100.0_f64).ln();
        let correction_factor = (corrections as f64 + 1.0).ln() / (500.0_f64).ln();
        let combination = (diversity_factor * 0.6 + correction_factor * 0.4).min(1.0);
        let remaining_gap = target - current;
        let round_gain = remaining_gap * 0.55 * combination.max(0.3);
        current = (current + round_gain).min(target);
        stats.accuracy_trajectory.push(current);
        stats.distillations_performed = round;
    }

    stats
}

/// Convenience: simulate with default base accuracy of 0.60.
pub fn simulate_distillation_default(corrections: usize, diversity: usize, rounds: usize) -> DistillationStats {
    simulate_distillation(corrections, diversity, rounds, 0.60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_correction(room: &str, tile: &str, confidence: f64) -> Correction {
        Correction::new(room, tile, "original", "expected", "cloud says expected", confidence)
    }

    // ── Correction tests ──

    #[test]
    fn correction_creation() {
        let c = Correction::new("room1", "tileA", "cat", "dog", "it's a dog", 0.95);
        assert_eq!(c.room_id, "room1");
        assert_eq!(c.tile_id, "tileA");
        assert_eq!(c.original_label, "cat");
        assert_eq!(c.expected_label, "dog");
        assert_eq!(c.cloud_response, "it's a dog");
        assert!((c.confidence - 0.95).abs() < 1e-9);
        assert_ne!(c.id, uuid::Uuid::nil());
    }

    #[test]
    fn confidence_clamped_high() {
        let c = Correction::new("r", "t", "a", "b", "c", 1.5);
        assert!((c.confidence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn confidence_clamped_low() {
        let c = Correction::new("r", "t", "a", "b", "c", -0.5);
        assert!((c.confidence - 0.0).abs() < 1e-9);
    }

    #[test]
    fn correction_serialization_roundtrip() {
        let c = Correction::new("room1", "tile1", "a", "b", "c", 0.8);
        let json = serde_json::to_string(&c).unwrap();
        let c2: Correction = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn unique_ids() {
        let c1 = Correction::new("r", "t", "a", "b", "c", 0.5);
        let c2 = Correction::new("r", "t", "a", "b", "c", 0.5);
        assert_ne!(c1.id, c2.id);
    }

    // ── Store tests ──

    #[test]
    fn store_add_and_len() {
        let mut store = CorrectionStore::new("room1");
        assert!(store.is_empty());
        store.add(make_correction("room1", "t1", 0.9));
        store.add(make_correction("room1", "t2", 0.8));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn store_empty_avg_confidence() {
        let store = CorrectionStore::new("room1");
        assert!((store.avg_confidence() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn store_avg_confidence() {
        let mut store = CorrectionStore::new("room1");
        store.add(make_correction("room1", "t1", 0.8));
        store.add(make_correction("room1", "t2", 1.0));
        assert!((store.avg_confidence() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn store_diversity() {
        let mut store = CorrectionStore::new("room1");
        store.add(make_correction("room1", "t1", 0.9));
        store.add(make_correction("room1", "t1", 0.8));
        store.add(make_correction("room1", "t2", 0.7));
        assert_eq!(store.diversity(), 2);
    }

    #[test]
    fn store_serialization_roundtrip() {
        let mut store = CorrectionStore::new("room1");
        store.add(make_correction("room1", "t1", 0.9));
        let json = serde_json::to_string(&store).unwrap();
        let s2: CorrectionStore = serde_json::from_str(&json).unwrap();
        assert_eq!(store, s2);
    }

    // ── Distillation trigger tests ──

    #[test]
    fn should_distill_below_threshold() {
        let mut store = CorrectionStore::new("room1");
        let config = DistillationConfig::new(10);
        for _ in 0..9 {
            store.add(make_correction("room1", "t1", 0.9));
        }
        assert!(!store.should_distill(&config));
    }

    #[test]
    fn should_distill_meets_threshold() {
        let mut store = CorrectionStore::new("room1");
        // Use a config with no time/interval requirement
        let config = DistillationConfig {
            min_corrections: 5,
            max_interval_secs: None, // skip time check
            min_diversity: Some(1),
        };
        for _ in 0..5 {
            store.add(make_correction("room1", "t1", 0.9));
        }
        assert!(store.should_distill(&config));
    }

    #[test]
    fn should_distill_fails_diversity() {
        let mut store = CorrectionStore::new("room1");
        let config = DistillationConfig {
            min_corrections: 3,
            max_interval_secs: None,
            min_diversity: Some(3),
        };
        // Only 2 unique tiles
        store.add(make_correction("room1", "t1", 0.9));
        store.add(make_correction("room1", "t1", 0.9));
        store.add(make_correction("room1", "t2", 0.9));
        assert!(!store.should_distill(&config));
    }

    #[test]
    fn should_distill_empty_store() {
        let store = CorrectionStore::new("room1");
        let config = DistillationConfig::new(1);
        assert!(!store.should_distill(&config));
    }

    // ── Export training data ──

    #[test]
    fn export_training_data_format() {
        let mut store = CorrectionStore::new("room1");
        store.add(Correction::new("room1", "tileA", "cat", "dog", "cloud", 0.9));
        let examples = store.export_training_data();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].input, "tile:room1:tileA");
        assert_eq!(examples[0].expected_output, "dog");
        assert!((examples[0].confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn export_training_data_empty() {
        let store = CorrectionStore::new("room1");
        assert!(store.export_training_data().is_empty());
    }

    #[test]
    fn training_example_serialization() {
        let ex = TrainingExample {
            input: "tile:room1:t1".into(),
            expected_output: "dog".into(),
            confidence: 0.95,
            tile_id: "t1".into(),
        };
        let json = serde_json::to_string(&ex).unwrap();
        let ex2: TrainingExample = serde_json::from_str(&json).unwrap();
        assert_eq!(ex, ex2);
    }

    // ── Simulation tests ──

    #[test]
    fn simulation_starts_near_base() {
        let stats = simulate_distillation_default(100, 10, 5);
        let first = stats.accuracy_trajectory[0];
        assert!(first > 0.60, "first round should improve from base");
        assert!(first < 0.80, "first round shouldn't overshoot");
    }

    #[test]
    fn simulation_converges_to_95_plus() {
        let stats = simulate_distillation_default(200, 20, 5);
        let final_acc = *stats.accuracy_trajectory.last().unwrap();
        assert!(
            final_acc >= 0.95,
            "after 5 rounds should converge to 95%+, got {:.3}",
            final_acc
        );
    }

    #[test]
    fn simulation_monotonic_improvement() {
        let stats = simulate_distillation_default(100, 10, 5);
        for window in stats.accuracy_trajectory.windows(2) {
            assert!(
                window[1] >= window[0] - 1e-9,
                "accuracy should not decrease: {:.4} -> {:.4}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn simulation_diminishing_returns() {
        let stats = simulate_distillation_default(100, 10, 5);
        let gains: Vec<f64> = stats
            .accuracy_trajectory
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        for window in gains.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-9,
                "gains should diminish: {:.4} -> {:.4}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn autonomy_percentage() {
        let mut stats = DistillationStats::new();
        assert!((stats.autonomy_percentage() - 0.0).abs() < 1e-9);
        stats.accuracy_trajectory.push(0.85);
        assert!((stats.autonomy_percentage() - 85.0).abs() < 1e-9);
    }

    // ── Config validation ──

    #[test]
    fn config_valid() {
        let config = DistillationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_zero_corrections_fails() {
        let config = DistillationConfig::new(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_zero_interval_fails() {
        let config = DistillationConfig::new(5).with_interval(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_zero_diversity_fails() {
        let config = DistillationConfig::new(5).with_min_diversity(0);
        assert!(config.validate().is_err());
    }

    // ── by_tile grouping ──

    #[test]
    fn by_tile_grouping() {
        let mut store = CorrectionStore::new("room1");
        store.add(make_correction("room1", "t1", 0.9));
        store.add(make_correction("room1", "t2", 0.8));
        store.add(make_correction("room1", "t1", 0.7));
        let groups = store.by_tile();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["t1"].len(), 2);
        assert_eq!(groups["t2"].len(), 1);
    }

    // ── Stats serialization ──

    #[test]
    fn stats_serialization_roundtrip() {
        let stats = simulate_distillation_default(50, 5, 3);
        let json = serde_json::to_string(&stats).unwrap();
        let s2: DistillationStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, s2);
    }

    // ── Duplicate corrections ──

    #[test]
    fn duplicate_corrections_both_stored() {
        let mut store = CorrectionStore::new("room1");
        let c = make_correction("room1", "t1", 0.9);
        store.add(c.clone());
        store.add(c);
        assert_eq!(store.len(), 2);
    }
}
