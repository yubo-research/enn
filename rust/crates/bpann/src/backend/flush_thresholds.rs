use super::BpannBackend;

impl BpannBackend {
    pub fn with_pending_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_flush_threshold = threshold;
        if self.pending_hard_flush_threshold < threshold {
            self.pending_hard_flush_threshold = threshold;
        }
        self
    }

    pub fn with_pending_hard_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_hard_flush_threshold = threshold.max(self.pending_flush_threshold);
        self
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.pending_flush_threshold
    }

    pub fn pending_hard_flush_threshold(&self) -> usize {
        self.pending_hard_flush_threshold
    }

    /// Update soft/hard pending flush thresholds (keeps `hard >= soft`).
    pub fn reconfigure_flush_thresholds(&mut self, soft: usize, hard: usize) {
        let soft = soft.max(1);
        self.pending_flush_threshold = soft;
        self.pending_hard_flush_threshold = hard.max(soft);
    }
}
