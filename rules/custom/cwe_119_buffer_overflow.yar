rule CWE119_BufferOverflow_strcpy
{
    meta:
        cwe         = "CWE-119"
        severity    = "HIGH"
        description = "Unsafe strcpy usage without bounds checking"

    strings:
        $unsafe1 = "strcpy(" ascii
        $unsafe2 = "strcat(" ascii
        $unsafe3 = "gets(" ascii
        $unsafe4 = "sprintf(" ascii
        $safe1   = "strncpy(" ascii
        $safe2   = "strncat(" ascii
        $safe3   = "snprintf(" ascii

    condition:
        any of ($unsafe*) and not any of ($safe*)
}

rule CWE119_BufferOverflow_memcpy
{
    meta:
        cwe         = "CWE-119"
        severity    = "HIGH"
        description = "Potentially unsafe memcpy without size validation"

    strings:
        $memcpy  = "memcpy(" ascii
        $malloc  = "malloc(" ascii

    condition:
        $memcpy and $malloc
}

rule CWE119_OffByOne_LenPlusOne
{
    meta:
        cwe         = "CWE-119"
        severity    = "HIGH"
        description = "memcpy with len+1 may write beyond buffer"

    strings:
        $memcpy  = "memcpy(" ascii
        $len1    = "len + 1" ascii
        $len2    = "length + 1" ascii
        $len3    = "size + 1" ascii
        $dest    = "dest" ascii
        $buf     = "buffer" ascii

    condition:
        $memcpy and
        any of ($len*) and
        any of ($dest, $buf)
}