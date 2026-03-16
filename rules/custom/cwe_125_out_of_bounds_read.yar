rule CWE125_OOBRead_ArrayAccess
{
    meta:
        cwe         = "CWE-125"
        severity    = "HIGH"
        description = "Array access without bounds validation"

    strings:
        $array   = /\w+\s*\[\s*\w+\s*\]/ ascii
        $malloc  = "malloc(" ascii
        $nocheck = /if\s*\(\s*\w+\s*[<>]=?\s*\d+\s*\)/ ascii

    condition:
        $array and $malloc and not $nocheck
}

rule CWE125_OOBRead_PointerArithmetic
{
    meta:
        cwe         = "CWE-125"
        severity    = "MEDIUM"
        description = "Unsafe pointer arithmetic that may read out of bounds"

    strings:
        $ptr1   = "*(ptr +" ascii
        $ptr2   = "*(buf +" ascii
        $ptr3   = "*(p +" ascii
        $check1 = "sizeof(" ascii
        $check2 = "strlen(" ascii

    condition:
        any of ($ptr*) and not any of ($check*)
}