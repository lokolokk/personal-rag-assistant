# Builds the first-boot script (cloud-init) from the files in deploy/aws/.
# Windows checkouts may turn line endings into CRLF, which breaks shell scripts on Linux,
# so every file is normalised to LF before it is embedded.

locals {
  aws_dir      = "${path.module}/../deploy/aws"
  ecr_registry = split("/", aws_ecr_repository.backend.repository_url)[0]

  user_data = templatefile("${path.module}/user_data.yaml.tftpl", {
    rag_conf_b64 = base64encode(templatefile("${local.aws_dir}/rag.conf.tftpl", {
      region           = var.region
      ecr_registry     = local.ecr_registry
      ecr_repository   = aws_ecr_repository.backend.name
      compose_version  = var.docker_compose_version
      compose_sha256   = var.docker_compose_sha256
      autostop_minutes = var.autostop_minutes
    }))
    daemon_json_b64 = base64encode(replace(file("${local.aws_dir}/daemon.json"), "\r\n", "\n"))
    service_b64     = base64encode(replace(file("${local.aws_dir}/rag-stack.service"), "\r\n", "\n"))
    compose_b64     = base64encode(replace(file("${path.module}/../compose.aws.yaml"), "\r\n", "\n"))
    nginx_b64       = base64encode(replace(file("${path.module}/../deploy/live/nginx.conf"), "\r\n", "\n"))
    rag_up_b64      = base64encode(replace(file("${local.aws_dir}/rag-up.sh"), "\r\n", "\n"))
    rag_down_b64    = base64encode(replace(file("${local.aws_dir}/rag-down.sh"), "\r\n", "\n"))
    bootstrap_b64   = base64encode(replace(file("${local.aws_dir}/bootstrap.sh"), "\r\n", "\n"))
  })
}
