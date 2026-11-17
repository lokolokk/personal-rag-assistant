variable "region" {
  description = "AWS region. The Free-plan project is locked to Sydney."
  type        = string
  default     = "ap-southeast-2"
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "rag"
}

variable "instance_type" {
  description = "EC2 type. t3.small: x86_64, 2 vCPU, 2 GiB, Free-tier eligible in Sydney."
  type        = string
  default     = "t3.small"
}

variable "root_volume_gb" {
  description = "Root (data) volume size in GiB. Holds Docker images and the live storage volume."
  type        = number
  default     = 30
}

variable "snapshot_retention" {
  description = "How many weekly EBS snapshots to keep."
  type        = number
  default     = 4
}

variable "autostop_minutes" {
  description = "The server shuts itself down this long after the stack starts (safety net)."
  type        = number
  default     = 180
}

variable "ngrok_domain" {
  description = "Fixed public ngrok domain (public, already in the frontend CSP). Not a secret."
  type        = string
  default     = "ebook-overcook-subsystem.ngrok-free.dev"
}

variable "frontend_origin" {
  description = "CORS origins as a JSON list: the Cloudflare Pages site. Not a secret."
  type        = string
  default     = "[\"https://personal-rag-assistant-ymwon04.pages.dev\"]"
}

variable "docker_compose_version" {
  description = "Docker Compose plugin release installed on the server."
  type        = string
  default     = "v5.7.1"
}

variable "docker_compose_sha256" {
  description = "SHA-256 of docker-compose-linux-x86_64 for that release (GitHub asset digest)."
  type        = string
  default     = "af426722a8af5de812fcee30cc1c39f0639f5f723cbadefb25e5d4db054064cf"
}

variable "create_github_users" {
  description = "Create the two least-privilege IAM users GitHub Actions will use. Set false if the account blocks IAM users."
  type        = bool
  default     = true
}
