rule CWE476_NullPointer_NoCheck
{
    meta:
        cwe         = "CWE-476"
        severity    = "MEDIUM"
        description = "Pointer used without NULL check after allocation"

    strings:
        $alloc1     = "malloc(" ascii
        $alloc2     = "calloc(" ascii
        $alloc3     = "realloc(" ascii
        $null_check1 = "== NULL" ascii
        $null_check2 = "!= NULL" ascii
        $null_check3 = "if (!ptr)" ascii

    condition:
        any of ($alloc*) and not any of ($null_check*)
}

rule CWE476_NullPointer_ReturnNoCheck
{
    meta:
        cwe         = "CWE-476"
        severity    = "MEDIUM"
        description = "Return value of function used without NULL check"

    strings:
        $func1  = "fopen(" ascii
        $func2  = "popen(" ascii
        $func3  = "opendir(" ascii
        $check1 = "== NULL" ascii
        $check2 = "!= NULL" ascii

    condition:
        any of ($func*) and not any of ($check*)
}