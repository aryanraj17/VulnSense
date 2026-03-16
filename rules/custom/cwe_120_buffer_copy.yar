rule CWE120_BufferCopy_NoBoundsCheck
{
    meta:
        cwe         = "CWE-120"
        severity    = "HIGH"
        description = "Buffer copy without checking size of input"

    strings:
        $unsafe1 = "strcpy(" ascii
        $unsafe2 = "wcscpy(" ascii
        $unsafe3 = "_tcscpy(" ascii
        $safe1   = "strncpy(" ascii
        $safe2   = "wcsncpy(" ascii
        $safe3   = "strlcpy(" ascii

    condition:
        any of ($unsafe*) and not any of ($safe*)
}

rule CWE120_BufferCopy_scanf
{
    meta:
        cwe         = "CWE-120"
        severity    = "MEDIUM"
        description = "Unbounded scanf reading into fixed buffer"

    strings:
        $scanf1  = "scanf(" ascii
        $scanf2  = "fscanf(" ascii
        $format  = "\"%s\"" ascii

    condition:
        any of ($scanf*) and $format
}