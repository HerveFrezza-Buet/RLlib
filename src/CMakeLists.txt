# Generate the list of source files
# using GLOB does not RECURSE

file(
	GLOB
	headers_files
	*.hpp
)

# Installation targets
install (FILES ${headers_files}
	 DESTINATION ${INCLUDE_INSTALL_DIR}
	 COMPONENT devel)
