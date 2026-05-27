output "vm_external_ip" {
  description = "ML Worker VM의 외부 IP"
  value       = google_compute_instance.ml_worker.network_interface[0].access_config[0].nat_ip
}

output "vm_internal_ip" {
  description = "ML Worker VM의 내부 IP"
  value       = google_compute_instance.ml_worker.network_interface[0].network_ip
}

output "db_connection_name" {
  description = "Cloud SQL 연결 이름"
  value       = google_sql_database_instance.ml_db.connection_name
}

output "db_ip" {
  description = "Cloud SQL IP 주소"
  value       = google_sql_database_instance.ml_db.ip_address[0].ip_address
}

output "gcs_bucket_name" {
  description = "GCS 버킷 이름"
  value       = google_storage_bucket.ml_artifacts.name
}

output "service_account_email" {
  description = "ML Worker 서비스 계정 이메일"
  value       = google_service_account.ml_worker.email
}

output "db_password" {
  description = "Cloud SQL ml_user 비밀번호"
  value       = random_password.db_password.result
  sensitive   = true
}
