rule CWE190_IntegerOverflow_MallocMultiply
{
    meta:
        cwe         = "CWE-190"
        severity    = "HIGH"
        description = "Multiplication result used directly in malloc without overflow check"

    strings:
        $multiply = /malloc\s*\([^)]*\*[^)]*\)/ ascii
        $safe1    = "SIZE_MAX" ascii
        $safe2    = "INT_MAX" ascii
        $safe3    = "overflow" nocase ascii

    condition:
        $multiply and not any of ($safe*)
}

rule CWE190_IntegerOverflow_Arithmetic
{
    meta:
        cwe         = "CWE-190"
        severity    = "HIGH"
        description = "Arithmetic on untrusted input without validation"

    strings:
        $input1 = "argv" ascii
        $input2 = "atoi(" ascii
        $input3 = "strtol(" ascii
        $arith1 = " * " ascii
        $arith2 = " + " ascii
        $check1 = "INT_MAX" ascii
        $check2 = "UINT_MAX" ascii

    condition:
        any of ($input*) and any of ($arith*) and not any of ($check*)
}

rule CWE190_MallocMultiply_Direct
{
    meta:
        cwe         = "CWE-190"
        severity    = "HIGH"
        description = "Direct multiplication used as malloc size argument"

    strings:
        $malloc  = "malloc(" ascii
        $mult1   = " * " ascii
        $memset  = "memset(" ascii

    condition:
        $malloc and $mult1 and $memset
}