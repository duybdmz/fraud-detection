terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
  }
}

provider "aws" {
  region = "ap-southeast-1"
}

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

variable "databricks_host" {
  type      = string
  sensitive = true
}

variable "databricks_token" {
  type      = string
  sensitive = true
}

resource "aws_s3_bucket" "fraud_demo" {
  bucket = "fraud-demo-434481793703"
  tags   = { Project = "fraud-detection-demo" }
}

resource "aws_s3_bucket_public_access_block" "fraud_demo" {
  bucket                  = aws_s3_bucket.fraud_demo.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# IAM role for Databricks cluster to access S3
resource "aws_iam_role" "databricks_s3" {
  name = "fraud-demo-databricks-s3"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
        Action    = "sts:AssumeRole"
      },
      {
        # Databricks Unity Catalog cross-account role
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::414351767826:role/unity-catalog-prod-UCMasterRole-14S5ZJVKOTYTL"
        }
        Action    = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "sts:ExternalId" = "434481793703"
          }
        }
      }
    ]
  })

  tags = { Project = "fraud-detection-demo" }
}

resource "aws_iam_role_policy" "databricks_s3" {
  name = "fraud-demo-s3-access"
  role = aws_iam_role.databricks_s3.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ]
      Resource = [
        aws_s3_bucket.fraud_demo.arn,
        "${aws_s3_bucket.fraud_demo.arn}/*"
      ]
    }]
  })
}

output "iam_role_arn" {
  value = aws_iam_role.databricks_s3.arn
}

# Databricks Storage Credential — dùng IAM role để truy cập S3
resource "databricks_storage_credential" "s3" {
  name = "fraud-demo-s3-credential"

  aws_iam_role {
    role_arn = aws_iam_role.databricks_s3.arn
  }
}

# External Location — trỏ vào S3 bucket
resource "databricks_external_location" "fraud_demo" {
  name            = "fraud-demo-s3"
  url             = "s3://${aws_s3_bucket.fraud_demo.id}"
  credential_name = databricks_storage_credential.s3.name
}

output "external_location_url" {
  value = databricks_external_location.fraud_demo.url
}
