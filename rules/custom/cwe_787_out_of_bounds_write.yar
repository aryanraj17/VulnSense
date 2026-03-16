rule CWE787_OOBWrite_memset
{
    meta:
        cwe         = "CWE-787"
        severity    = "HIGH"
        description = "memset writing beyond allocated buffer size"

    strings:
        $memset  = "memset(" ascii
        $malloc  = "malloc(" ascii
        $sizeof  = "sizeof(" ascii

    condition:
        $memset and $malloc and not $sizeof
}

rule CWE787_OOBWrite_strcpy
{
    meta:
        cwe         = "CWE-787"
        severity    = "HIGH"
        description = "Unbounded write into fixed size buffer"

    strings:
        $unsafe1 = "strcpy(" ascii
        $unsafe2 = "sprintf(" ascii
        $unsafe3 = "gets(" ascii
        $buf1    = "char " ascii
        $buf2    = "wchar_t " ascii

    condition:
        any of ($unsafe*) and any of ($buf*)
}