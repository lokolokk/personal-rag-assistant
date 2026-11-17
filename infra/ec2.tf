resource "aws_instance" "server" {
  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.server.id]
  iam_instance_profile        = aws_iam_instance_profile.server.name
  associate_public_ip_address = true

  # First-boot setup (Docker, swap, systemd, the stack). cloud-init runs it only once per
  # instance, so a changed script replaces the instance. Fine now (no data yet). BEFORE real
  # data lives on the disk, add user_data_base64 to ignore_changes below so edits never wipe it.
  user_data_base64            = base64gzip(local.user_data)
  user_data_replace_on_change = true

  # A shutdown from inside the OS (the 3 h auto-stop planned for P2) stops the instance, not deletes it.
  instance_initiated_shutdown_behavior = "stop"

  # "standard" credit mode: when burst credits run out the CPU is throttled instead of billing
  # surplus credits. Hard cost control for an on-demand demo box.
  credit_specification {
    cpu_credits = "standard"
  }

  # IMDSv2 only. Containers cannot reach the instance credentials (hop limit 1).
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    # DLM snapshots every volume carrying this tag.
    tags = {
      Name     = "${var.name_prefix}-server-root"
      Snapshot = "weekly"
    }
  }

  tags = { Name = "${var.name_prefix}-server" }

  lifecycle {
    # A newer AMI must not replace the server (and its data). Update the OS in place.
    # Also ignore the public-IP flag: a STOPPED instance reports no public IP, which would
    # otherwise look like a change and force a replacement on every plan.
    ignore_changes = [ami, associate_public_ip_address]
  }
}
