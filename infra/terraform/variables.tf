variable "project_id" {
  type        = string
  description = "GCP 프로젝트 ID"
}

variable "region" {
  type        = string
  description = "GCP 리전"
  default     = "asia-northeast3"
}

variable "zone" {
  type        = string
  description = "GCP 존"
  default     = "asia-northeast3-a"
}

variable "machine_type" {
  type        = string
  description = "GCE VM 머신 타입"
  default     = "e2-standard-4"
}

variable "gpu_enabled" {
  type        = bool
  description = "GPU 사용 여부"
  default     = false
}

variable "gpu_type" {
  type        = string
  description = "GPU 가속기 타입"
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  type        = number
  description = "GPU 개수"
  default     = 1
}

variable "db_tier" {
  type        = string
  description = "Cloud SQL 인스턴스 티어"
  default     = "db-f1-micro"
}

variable "ssh_source_ranges" {
  type        = list(string)
  description = "SSH 접속 허용 소스 IP 범위"
  default     = ["0.0.0.0/0"]
}
