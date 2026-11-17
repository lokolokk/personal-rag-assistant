# No inbound rules at all: no SSH, no open ports. Admin access is SSM Session Manager and the
# public address is an outbound ngrok tunnel started on the instance.
resource "aws_security_group" "server" {
  name        = "${var.name_prefix}-server"
  description = "RAG server: no inbound, HTTPS-only outbound"
  vpc_id      = data.aws_vpc.default.id

  tags = { Name = "${var.name_prefix}-server" }
}

# Outbound HTTPS (ECR, SSM, Qwen API, ngrok, OS package repos). DNS uses the VPC resolver and
# needs no rule. Widen only when a real boot proves something else is required.
resource "aws_vpc_security_group_egress_rule" "https_out" {
  security_group_id = aws_security_group.server.id
  description       = "HTTPS to the internet"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

# Port 80 outbound, proven necessary on the first real boot: the ngrok agent downloads its
# certificate revocation list (CRL) over plain HTTP and cannot connect without it. Outbound only;
# there is still no inbound rule.
resource "aws_vpc_security_group_egress_rule" "http_out" {
  security_group_id = aws_security_group.server.id
  description       = "HTTP to the internet (ngrok certificate revocation list)"
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  cidr_ipv4         = "0.0.0.0/0"
}
