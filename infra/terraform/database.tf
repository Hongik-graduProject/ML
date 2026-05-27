resource "random_password" "db_password" {
  length           = 20
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "google_sql_database_instance" "ml_db" {
  name             = "ml-db"
  database_version = "POSTGRES_15"
  region           = var.region

  deletion_protection = false

  settings {
    tier = var.db_tier

    ip_configuration {
      ipv4_enabled = true

      authorized_networks {
        name  = "ml-worker-vm"
        value = "${google_compute_instance.ml_worker.network_interface[0].access_config[0].nat_ip}/32"
      }
    }

    backup_configuration {
      enabled = true
    }
  }

  depends_on = [google_compute_instance.ml_worker]
}

resource "google_sql_database" "theme_classifier" {
  name     = "theme_classifier"
  instance = google_sql_database_instance.ml_db.name
}

resource "google_sql_user" "ml_user" {
  name     = "ml_user"
  instance = google_sql_database_instance.ml_db.name
  password = random_password.db_password.result
}
