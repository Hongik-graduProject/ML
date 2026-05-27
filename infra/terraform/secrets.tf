resource "google_service_account" "ml_worker" {
  account_id   = "ml-worker"
  display_name = "ML Worker SA"
}

resource "google_project_iam_member" "ml_worker_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_worker.email}"
}

resource "google_project_iam_member" "ml_worker_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.ml_worker.email}"
}

resource "google_project_iam_member" "ml_worker_secretmanager" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.ml_worker.email}"
}

resource "google_secret_manager_secret" "dart_api_key" {
  secret_id = "dart-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "dart_api_key_accessor" {
  secret_id = google_secret_manager_secret.dart_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ml_worker.email}"
}
