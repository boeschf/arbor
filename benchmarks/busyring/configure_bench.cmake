
#set(OUTFILE "tmp1.txt" CACHE STRING "")

configure_file(
    ${INFILE}
    ${OUTFILE}
    @ONLY
    NEWLINE_STYLE UNIX)
