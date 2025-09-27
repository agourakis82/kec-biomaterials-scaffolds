# =============================================================================
# DARWIN Infrastructure - Terraform Versions Configuration
# =============================================================================

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    
    google-beta = {
      source  = "hashicorp/google-beta" 
      version = "~> 5.0"
    }
    
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
    
    time = {
      source  = "hashicorp/time"
      version = "~> 0.9"
    }
    
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
    
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    
    template = {
      source  = "hashicorp/template"
      version = "~> 2.2"
    }
    
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
  
  # Provider feature flags
  provider_meta "google" {
    module_name = "darwin-infrastructure"
  }
  
  provider_meta "google-beta" {
    module_name = "darwin-infrastructure-beta"
  }
}

# =============================================================================
# Provider Configuration with Enhanced Features
# =============================================================================

# Main Google Cloud provider
# provider "google" {
#   project = var.project_id
#   region  = var.region
#   zone    = "${var.region}-a"
#
#   # Request timeout
#   request_timeout = "60s"
#
#   # Batching configuration
#   batching {
#     send_after      = "10s"
#     enable_batching = true
#   }
#
#   # User project override
#   user_project_override = true
#
#   # Default labels for all resources
#   default_labels = {
#     environment    = var.environment
#     project        = var.project_name
#     managed_by     = "terraform"
#     cost_center    = "research"
#     team           = "kec-biomaterials"
#     deployment_id  = formatdate("YYYYMMDD-hhmm", timestamp())
#   }
# }
#
# # Google Beta provider for preview features
# provider "google-beta" {
#   project = var.project_id
#   region  = var.region
#   zone    = "${var.region}-a"
#
#   # Request timeout
#   request_timeout = "60s"
#
#   # Batching configuration
#   batching {
#     send_after      = "10s"
#     enable_batching = true
#   }
#
#   # User project override
#   user_project_override = true
#
#   # Default labels for all resources
#   default_labels = {
#     environment    = var.environment
#     project        = var.project_name
#     managed_by     = "terraform"
#     cost_center    = "research"
#     team           = "kec-biomaterials"
#     deployment_id  = formatdate("YYYYMMDD-hhmm", timestamp())
#   }
# }

# Random provider for generating secure values
provider "random" {
  # No configuration needed
}

# Time provider for time-based resources
provider "time" {
  # No configuration needed
}

# Null provider for running local commands
provider "null" {
  # No configuration needed
}

# Local provider for local file operations
provider "local" {
  # No configuration needed
}

# Template provider for template rendering
provider "template" {
  # No configuration needed
}

# Archive provider for creating archives
provider "archive" {
  # No configuration needed
}