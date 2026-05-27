data "google_compute_image" "ubuntu_2204" {
  family  = "ubuntu-2204-lts"
  project = "ubuntu-os-cloud"
}

resource "google_compute_instance" "ml_worker" {
  name         = "ml-worker"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["ml-worker", "ssh"]

  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu_2204.self_link
      size  = 50
      type  = "pd-ssd"
    }
  }

  dynamic "guest_accelerator" {
    for_each = var.gpu_enabled ? [1] : []
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }

  scheduling {
    on_host_maintenance = var.gpu_enabled ? "TERMINATE" : "MIGRATE"
    automatic_restart   = var.gpu_enabled ? false : true
  }

  network_interface {
    subnetwork = google_compute_subnetwork.ml_subnet.id

    access_config {
      # 외부 IP 자동 할당
    }
  }

  service_account {
    email  = google_service_account.ml_worker.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    apt-get update -y
  EOT
}
