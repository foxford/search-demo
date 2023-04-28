{{/*
Expand the name of the chart.
*/}}
{{- define "search-demo.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "search-demo.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "search-demo.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}} # TODO
{{- define "search-demo.labels" -}}
helm.sh/chart: {{ include "search-demo.chart" . }}
app.kubernetes.io/version: {{ .Values.app.image.tag | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{ include "search-demo.selectorLabels" . }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "search-demo.selectorLabels" -}}
app.kubernetes.io/name: {{ include "search-demo.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Tenant Service Audience
*/}}
{{- define "search-demo.tenantServiceAudience" -}}
{{- $tenant := . -}}
{{- list "svc" $tenant | join "." -}}
{{- end -}}

{{/*
Tenant User Audience
*/}}
{{- define "search-demo.tenantUserAudience" -}}
{{- $tenant := . -}}
{{- list "usr" $tenant | join "." -}}
{{- end -}}

{{/*
Tenant Object Audience
*/}}
{{- define "search-demo.tenantObjectAudience" -}}
{{- $namespace := index . 0 -}}
{{- $tenant := index . 1 -}}
{{- $env := regexSplit "-" $namespace -1 | first -}}
{{- $devEnv := ""}}
{{- if ne $env "p" }}
{{- $devEnv = regexReplaceAll "(s)(\\d\\d)" $env "staging${2}" }}
{{- $devEnv = regexReplaceAll "(t)(\\d\\d)" $devEnv "testing${2}" }}
{{- end }}
{{- list $devEnv $tenant | compact | join "." }}
{{- end }}

{{/*
Namespace in ingress path.
converts as follows:
- testing01 -> t01
- staging01-search-demo-ng -> s01/search-demo-ng
- production-search-demo-ng -> search-demo-ng
*/}}
{{- define "search-demo.ingressPathNamespace" -}}
{{- $ns_head := regexSplit "-" .Release.Namespace -1 | first }}
{{- $ns_tail := regexSplit "-" .Release.Namespace -1 | rest | join "-" }}
{{- if has $ns_head (list "production" "p") }}
{{- regexReplaceAll "(.*)-ng(.*)" $ns_tail "${1}-foxford${2}" }}
{{- else }}
{{- $v := list (regexReplaceAll "(.)[^\\d]*(.+)" $ns_head "${1}${2}") $ns_tail | compact | join "/" }}
{{- regexReplaceAll "(.*)-ng(.*)" $v "${1}-foxford${2}" }}
{{- end }}
{{- end }}

{{/*
Ingress path.
*/}}
{{- define "search-demo.ingressPath" -}}
{{- $shortns := regexSplit "-" .Release.Namespace -1 | first }}
{{- list "" (include "search-demo.ingressPathNamespace" .) (include "search-demo.name" .) | join "/" }}
{{- end }}

{{/*
Create volumeMount name from audience and secret name
*/}}
{{- define "search-demo.volumeMountName" -}}
{{- $audience := index . 0 -}}
{{- $secret := index . 1 -}}
{{- printf "%s-%s-secret" $audience $secret | replace "." "-" | trunc 63 | trimSuffix "-" }}
{{- end }}
